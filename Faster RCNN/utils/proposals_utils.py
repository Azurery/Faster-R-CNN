import tensorflow as tf
from utils.bbox_utils import encode_bboxes, decode_bboxes, clip_bboxes, filter_overlap_bboxes, pairwise_iou
from keras.layers import Conv2D
from keras.regularizers import l2

class rpn_head_creator(tf.keras.Model):
	'''
				features
					|
					|
			  3 * 3 conv, 512
			        |
					|
			-----------------------
			|					  |
		1 *	1 conv, 18       1 * 1 conv, 36

	rpn_head 主要是位于 RPN 网络的头部，包含两个分支：
	1. cls layer：用于产生 `2k` 个 scores
		即由 `2k` 个 `1*1` filter 组成
	
	2. reg layer：用于产生 `4k` 个 coordinates
		即由 `4k` 个 `1*1` filter 组成

	【注】`k` 的取值为 `9`
	'''

	def __init__(self, num_anchors, weight_decay):
		super().__init__()
		self._num_anchors = num_anchors
		self._rpn_conv = Conv2D(512, [3, 3], 
								padding='same',
								kernel_initializer=tf.random_normal_initializer(0, 0.01),
								kernel_regularizer=l2(weight_decay), 
								name='rpn_conv1')

		self._rpn_score_conv = Conv2D(self._num_anchors * 2, [1, 1], 
									kernel_initializer=tf.random_normal_initializer(0, 0.01),
									kernel_regularizer=l2(weight_decay), 
									name='rpn_score_conv')
		
		self._rpn_coordinate_conv = Conv2D(self._num_anchors * 4, [1, 1],
										kernel_initializer=tf.random_normal_initializer(0, 0.01),
										kernel_regularizer=l2(weight_decay),
										name='rpn_coordinate_conv')
	
	def call(self, features):
		# 先进行一个 `3*3` conv
		rpn_conv1 = self._rpn_conv(features)

		'''
		`rpn_socres` shape：`[H * W * 18]`
		Q：为什么要在 softmax 的前后要各加入一个 reshape？
		A: 因为 softmax 的主要目的是：用于区分 fg/bg，即是一个二分类问题

		由于 TF 中的数据形式为 `[N, H, W, C]`，所以需要将 `C` 这个维度进行划分，即
		`[H * W * 18]` ---> `[H * W, 18]` ---> `[H * W, 9, 2]`
		'''
		rpn_scores = self._rpn_score_conv(rpn_conv1)
		
		# `[H * W * 18]` ---> `[H * W, 18]`
		rpn_scores = tf.reshape(rpn_scores, [-1, self._num_anchors * 2])
		
		# `[H * W, 18]` ---> `[H * W, 2, 9]`
		rpn_scores = tf.reshape(rpn_scores, [-1, 2, self._num_anchors])

		# `[H * W, 2, 9]` ---> `[H * W, 9, 2]`
		rpn_scores = tf.transpose(rpn_scores, [0, 2, 1])
		
		# `[H * W, 9, 2]` ---> `[H * W * 9, 2]`
		rpn_scores = tf.reshape(rpn_scores, [-1, 2])

		#  ---> `[H * W * 9, 2]`
		rpn_scores = tf.nn.softmax(rpn_scores)
		rpn_scores = tf.reshape(rpn_scores, [-1, self._num_anchors, 2])

		# `[H * W * 9, 2]`---> `[H * W, 2, 9]`
		rpn_scores = tf.transpose(rpn_scores, [0, 2, 1])
		
		# `[H * W, 2, 9]`---> `[H * W, 18]`
		rpn_scores = tf.reshape(rpn_scores, [-1, 2 * self._num_anchors])

		# `[H * W, 18]`---> `[H * W * 18]`
		# 这个操作骚啊，没看懂
		rpn_scores = tf.reshape(rpn_scores[:, self._num_anchors:], [-1])

		rpn_coordinates = self._rpn_coordinate_conv(rpn_conv1)
		rpn_coordinates = tf.reshape(rpn_coordinates, [-1, 4])
		
		return rpn_scores, rpn_coordinates


'''
在 RPN 中，从上万个 anchors 中，选择一定数目（`2000` 或者 `300`），调整大小和位置，生成 RoIs，
用以 Faster R-CNN 训练或者测试。
'''
class rpn_proposals_creator(tf.keras.Model):
	
	def __init__(self, 
				num_anchors=9, 
				nms_threshold=0.7,
				num_pre_nms_train=12000,
				num_post_nms_train=2000,
				num_pre_nms_test=6000,
				num_post_nms_test=300):	
		super().__init__()
		self.num_anchors = num_anchors
		self.nms_threshold = nms_threshold
		self.num_pre_nms_train = num_pre_nms_train
		self.num_post_nms_train = num_post_nms_train
		self.num_pre_nms_test = num_pre_nms_test
		self.num_post_nms_test = num_post_nms_test


	def call(self, inputs, training=None):
		'''
        生成 rpn 的结果，即一组 bboxes，用于后续 roi pooling
        总体过程：
        1. 使用 anchors 和 rpn_pred 修正，获取所有预测结果。
        2. 对选中修正后的 anchors 进行处理（剪裁）。
        3. 根据 rpn_score 获取 num_pre_nms 个 anchors。
        4. 进行 nms。
        5. 根据 rpn_score 排序，获取 num_post_nms 个 anchors 作为 proposal 结果。
        '''

        # `rpn_coordinates` shape: `[num_anchors * feature_width * feature_height, 4]`
        # `anchors` shape: `[num_anchors * feature_width * feature_height, 4]`
        # `scores` shape: `[feature_width * feature_height * num_anchors,]`
        # `image_shape` shape: `[2, ]`

		rpn_scores, rpn_coordinates, anchors, image_shape = inputs
		# print(image_shape[1].numpy())
		# 1. 使用 anchors 和 rpn_pred 进行修正，得到解码后的 bbox 的坐标
		# `[num_anchors * feature_width * feature_height, 4]`
		decoded_bboxes = decode_bboxes(anchors, rpn_coordinates)

		# 对选中且解码后的 anchors 进行 clip 和 filter
		decoded_bboxes = clip_bboxes(decoded_bboxes, image_shape[0].numpy(), image_shape[1].numpy())

		num_nms_bbox = self.num_post_nms_train if training else self.num_post_nms_test
		selected_indices = tf.image.non_max_suppression(tf.cast(decoded_bboxes, dtype=tf.float32),
														rpn_scores,
														max_output_size=num_nms_bbox,
														iou_threshold=self.nms_threshold)
		selected_bboxes = tf.gather(decoded_bboxes, selected_indices)
		tf.logging.info('RPN net generates %d proposals' % tf.size(selected_indices))

		# 由于 `selected_bboxes` 不参与 training，则可以不让其进行 BP
		return tf.stop_gradient(selected_bboxes)


class rpn_target_anchors_createor(tf.keras.Model):
	'''
	作用：从 `20000` 多个候选的 anchors 选出 `256` 个 anchors 进行分类和回归位置。
	
	选择过程如下：
	1. 对于每一个 gt bbox，选择和它 IoU 最高的一个 anchor 作为正样本
	2. 对于剩下的 anchors，从中选择和任意一个 gt bbox `IoU > 0.7` 的 anchor，作为正样本，正样本的数目不超过 `128` 个。
	3. 随机选择和 gt bbox `IoU < 0.3` 的anchor作为负样本。负样本和正样本的总数为 `256`。
	'''

	def __init__(self, positive_iou_threshold=.7,
						negative_iou_threshold=.3,
						total_samples=256,
						max_positive_samples=128):
		
		super().__init__()
		self.positive_iou_threshold = positive_iou_threshold
		self.negative_iou_threshold = negative_iou_threshold
		self.total_samples = total_samples
		self.max_positive_samples = max_positive_samples

	
	def call(self, features, training=None):

		gt_bboxes, image_shape, anchors = features
		# 因为 `anchors` 的 shape 为：`[num_anchors, 4]` 
		# 下面这么做的目的是为了得到总的 anchors 数量
		num_anchors = anchors.get_shape().as_list()[0]

		# 1. 首先对 anchors 进行 filter，筛选符合边界要求的 anchors，之后的所有操作都是基于帅选后的结果
		tf.logging.info('Before filtering, the number of target anchors are %d' % anchors.shape[0])
		filtered_indicies, anchors = filter_overlap_bboxes(anchors, image_shape)
		tf.logging.info('After filtering, the number of target anchors are %d' % anchors.shape[0])

		labels, anchor_max_iou_indices, positive_indices = self._create_label(anchors, gt_bboxes)

		# 在进行 encode 时，需要根据先验框 anchors 和 gt boxes 进行转换
		# $l^{c x}=\left(b^{c x}-d^{c x}\right) / d^{w}, l^{c y}=\left(b^{c y}-d^{c y}\right) / d^{h}$
		# $l^{w}=\log \left(b^{w} / d^{w}\right), l^{h}=\log \left(b^{h} / d^{h}\right)$
		target_anchors = encode_bboxes(anchors, tf.gather(gt_bboxes, anchor_max_iou_indices))

		# 只有正样本才有 reg loss
		'''
		rpn 中的 loss 函数为：
			$\mathrm{L}\left(\left\{p_{i}\right\},\left\{t_{i}\right\}\right)=\frac{1}{N_{\mathrm{cls}}} \sum_{i} \mathrm{L}_{\mathrm{cls}}\left(p_{i}, p_{i}^{*}\right)+\lambda \frac{1}{N_{\mathrm{reg}}} \sum_{i} p_{i}^{*} \mathrm{L}_{\mathrm{reg}}\left(t_{i}, t_{i}^{*}\right)$
			
			即，L({Pi}, {ti}) = (1/Ncls) * \sum{Lcls(pi, pi*)} 
							  + λ * (1/Nreg) * \sum{pi* * Lreg(ti, ti*)}

		其中： 
		- `i` 表示 anchors index

		- `pi` 表示 positive predicted softmax probability，
			即，`pi` is the predicted probability of anchor `i` being an object.

		- `pi*` 表示 GT labels
		1. 当第 `i` 个 anchor 与 GT 间 `IoU > 0.7`，认为该 anchor 是 positive，即 `pi* = 1`
		2. 反之，`IoU < 0.3` 时，认为该 anchor 是 negative，即 `pi* = 0`
		3. 至于那些 `0.3 < IoU < 0.7` 的 anchor 则不参与训练

		- `ti` 表示 predicted bbox（即预测框）
			即，`ti` is a vector representing the `4` parameterized coordinates of the predicted bounding box

		- `ti*` 表示与 positive anchor 对应的 GT box
			即，`ti*` is that of the gt box associated with a positive anchor.


		1. The classification loss `Lcls` is log loss over two classes(object vs not object)
		2. The regression loss, `Lreg(ti, ti*) = R(ti - ti*)` where R is the robust loss function(smooth L1).
			The term `pi*Lreg` means the regression loss is activated only for positive anchors(`pi* = 1`) and
			is disabled otherwise(`pi* = 0`).

		可以看到，整个 Loss 分为两部分：
		- cls loss，即 rpn_cls_loss 层计算的 softmax loss
			用于分类 anchors 为 postive 与 negative 的网络训练
		- reg loss，即 rpn_loss_bbox 层计算的 soomth L1 loss
			用于 bbox regression 网络训练。
			【注意】在该 loss 中乘了 `pi*`，相当于只关心 positive anchors 的回归（其实,
			在回归中也完全没必要去关心 negative）。


		由于在实际过程中，`Ncls` 和 `Nreg` 差距过大，用参数 `λ` 平衡二者（如 `Ncls = 256`，`Nreg = 2400` 时，
		设置 `λ = Nreg / Ncls ≈ 10`，使总的网络 Loss 计算过程中能够均匀考虑两种 Loss。
		
		这里比较重要是 `Lreg` 使用的 soomth L1 loss，计算公式如下：
			Lreg(ti, ti*) = \sum {smooth_L1(ti - ti*)}
			
						|---- 0.5x^2      if |x| < 1 
			smooth_L1(x) = |
						|---- |x| - 0.5   otherwise
		'''		

		# 因为 `anchors` 的 shape 为：`[num_anchors, 4]` 
		# 下面的意思是：生成一个二维数组， 其中 `anchors.shape[0]` 表示 anchors 的总数，而 `4` 表示 `4` 个坐标
		# 所以，`pi_star` 这个二维数组表示有 `num_anchors` 个  anchors，每一个 anchor 的坐标为 `4` 个

		# pi_star 实际上中的就是 `pi*`，即正样本
		rpn_w_in = tf.zeros((anchors.shape[0], 4), dtype=tf.float32)
		rpn_w_in = tf.scatter_update(tf.Variable(rpn_w_in), 
									positive_indices, 1)

		# `N_reg` 指的就是上述式子中的 `Nreg`
		rpn_w_out = tf.zeros((anchors.shape[0], 4), dtype=tf.float32)
		
		# `N_reg` 是进行标准化操作，就是取平均。这个平均是把所有的 labels `0` 和 labels `1` 加起来。
		rpn_w_out = tf.scatter_update(tf.Variable(rpn_w_out),
									tf.where(labels >= 0)[:, 0],
									1.0 / tf.reduce_sum(tf.cast(labels >= 0, dtype=tf.float32)))

		return tf.stop_gradient(self._mapping(labels, num_anchors, filtered_indicies, -1)),\
				tf.stop_gradient(self._mapping(target_anchors, num_anchors, filtered_indicies, 0)),\
				tf.stop_gradient(self._mapping(rpn_w_in, num_anchors, filtered_indicies, 0)),\
				tf.stop_gradient(self._mapping(rpn_w_out, num_anchors, filtered_indicies, 0))


	def _create_label(self, anchors, gt_bboxes):
		'''
		作用：
			创建 label，其中，`1` 为正样本, `0` 为负样本, `-1` 则不管
		
		label 分为 3 类：
			- 一类是 `0`，即背景 label
			- 一类是 `1`，即前景 label
			- 另一类既不是前景也不是背景，置为 `-1`。
		论文中说只有前景和背景对训练目标有用，这种 `-1` 的 label 对训练没用。
		'''	

		# `anchors` 的 shape 为：`[num_anchors, 4]` 。
		# 注：一个 anchor 只能对应一个 gt box，但一个 gt box 可以对应多个 anchor
		# `labels` 这个 tensor 中存储的是每个 anchor 与 对应 gt box 的最大 IoU 值
		labels = tf.negative(tf.ones((anchors.shape[0],), tf.int32))
		
		'''
		作用：计算 IoU
			`iou` 的 shape：`[M, N]`
		'''
		iou = pairwise_iou(anchors, gt_bboxes)

		
		# 找出 某一个 anchor 与所有 gt_bboxes IoU 中最大的一个
		# 因为 `iou` 的 shape 为 `[M, N]`，为一个二维数组，则 `iou` 中的每一行就代表某一个 anchor 与 所有 `N` 个
		# gt_bboxes 的 IoU。这就需要找到 `iou` 每一行中的最大值
		anchor_max_iou_indices = tf.argmax(iou, axis=1, output_type=tf.int32)

		# 所以，`anchor_max_iou` 就是找到每一个 anchor 与 gt_bboxes 最大的 IoU
		anchor_max_iou = tf.reduce_max(iou, axis=1) 


		# FIXME：个人觉得这一步没有必要
		# # 所以，`gt_max_iou` 就是找到每一个 gt_bbox 与所有 anchors 最大的 IoU
		# gt_max_iou = tf.reduce_max(iou, axis=0)

		# # 找出 某一个 gt_bbox 与所有 anchors IoU 中最大的一个
		# # 因为 `iou` 的 shape 为 `[M, N]`，为一个二维数组，则 `iou` 中的每一列就代表与某一个 gt_bbox 相交的 `M` 个
		# # anchors 的 IoU。这就需要找到 `iou` 每一列中的最大值

		# gt_max_iou_indices = tf.where(tf.equal(iou, gt_max_iou))[:, 0]

		'''
		- 与 gt_bboxes 的 `max_iou > 0.7` 的 anchor为正样本----positive
		- `max_iou < 0.3` 的 anchor 为负样本-----negative
		
		其中，
		- positive_iou_threshold=.7,
		- negative_iou_threshold=.3
		'''
		negative_labels = anchor_max_iou < self.negative_iou_threshold
		positive_labels = anchor_max_iou >= self.positive_iou_threshold

		# 得到 negative 的位置
		labels = tf.where(negative_labels, tf.zeros_like(labels), labels)

		# 个人觉得这一步也没有必要
		# labels = tf.scatter_update(tf.Variable(labels), gt_max_iou_indices, 1)

		negative_indices = tf.where(tf.equal(labels, 0))[:, 0]
		
		# 得到 positive 的位置
		labels = tf.where(positive_labels, tf.ones_like(labels), labels)
		positive_indices = tf.where(tf.equal(labels, 1))[:, 0]

		'''
		`positive_indices` 将 `labels=1` 的 anchor 序号抽取出来，如果它的个数大于最大正样本数量，
		将随机抽取出一些 disable 掉，即将 `labels`设置为 `1`。

		通常，positive 框的数量远小于 negative 框，二者加起来一共 `256` 个。注意，现在总的 anchors
		的数量没有发生变化，labels 的大小也和总的 anchors 的数量一样，只不过其中标 `1` 和标 `0` 的数目
		加起来一共为 `256`，至于其他没有被考虑的 anchors 就直接标为 `-1`
		'''

		if tf.size(positive_indices) > self.max_positive_samples:
			# FIXME：没有对 IoU 进行排序？
			positive_indices = tf.random_shuffle(positive_indices)
			positive_indices = positive_indices[ :self.max_positive_samples]
			unuseful_indices = positive_indices[self.max_positive_samples: ]
			# 主要是用于更新 `labels`，使得 `labels` 中的 unuseful 设置为 `-1`
			labels = tf.scatter_update(tf.Variable(labels), unuseful_indices, -1)

		# 计算 positive 的数量
		num_positive = tf.reduce_sum(tf.cast(tf.equal(labels, 1), dtype=tf.int32))
		# 计算 negative 的数量
		num_negative = self.total_samples - num_positive

		# 不是很理解为什么 `[:, 0]` 就能得到 `negative_indices`？
		# 个人理解，因为 `anchors` 的 shape 为：`[num_anchors, 4]` ，
		# 其中，`anchors.shape[0]` 表示 anchors 的总数，而 `anchors.shape[1]` 表示 anchors 的 4 个坐标
		# 因为 `labels` 的 shape 为：`anchors.shape[0] = [num_anchors]`,
		negative_indices = tf.where(tf.equal(labels, 0))[:, 0]

		if tf.size(negative_indices) > num_negative:
			negative_indices = tf.random_shuffle(negative_indices)
			negative_indices = negative_indices[ :num_negative]
			unuseful_indices = negative_indices[num_negative: ]
			labels = tf.scatter_update(tf.Variable(labels), unuseful_indices, -1)

		tf.logging.info('target generate %d positive and %d negative.' % (tf.size(positive_indices), tf.size(negative_indices)))

		return labels, anchor_max_iou_indices, positive_indices

	# 不懂
	def _mapping(self, anchors, num_anchors, indices, filled_value=0):
		'''
		因为总的 anchors 裁减掉了 `2/3` 左右，仅仅保留在图像内的 anchors。这里就是将其复原作为下一层的输入了，并 reshape 成相应的格式。
		
		简而言之，就是将经过 filter 的 anchors映射回到原始的 anchors 中，其实主要就是 index 的转换
		'''
		
		if len(anchors.shape) == 1:
			mapping_ret = tf.ones([num_anchors], dtype=tf.float32) * filled_value
			mapping_ret = tf.scatter_update(tf.Variable(mapping_ret), indices, tf.cast(anchors, dtype=tf.float32))
		else:
			mapping_ret = tf.ones([num_anchors, ] + anchors.get_shape().as_list()[1:], dtype=tf.float32)
			mapping_ret = tf.scatter_update(tf.Variable(mapping_ret), indices, tf.cast(anchors, dtype=tf.float32))
		return mapping_ret

class roi_target_proposals_createor(tf.keras.Model):
	def __init__(self, num_classes=21, 
				positive_iou_threshold=0.5, negative_iou_threshold=0.5,
				num_samples=128, max_positive_samples=32):
		super().__init__()

		self._num_classes = num_classes
		self._positive_iou_threshold = positive_iou_threshold
		self._negative_iou_threshold = negative_iou_threshold
		self._num_samples = num_samples
		self._max_positive_samples = max_positive_samples

	def call(self, inputs, training=None):
		'''
		这一步的主要目的：生成用于训练的 roi proposals

		总体过程：
			1. 计算 roi proposals 与 gt boxes 的 IoU
			2. 设置与 gt_boxes 的 `max_iou > 0.5` 的 roi proposals 为 positive
				与 gt_boxes 的 `max_iou < 0.5` 的 roi proposals 为 negative
			3. 再缩减 positive 和 negative 的数量：
				- positive 的数量不超过 `max_positive_samples`
				- positive + negative 的总数不超过 `num_samples`
				- 如果 negative 的数量过少，就随机进行填充
			4. 最终输出 `5` 个结果：
				- roi proposals `[128, 4]`
				- 每个 roi proposal 对应的 label `[128,]`
					1. 如果为 `0`，则表示 negative
					2. 如果为 `>1`，则表示 positive
				- 每个 roi proposal 对应的 coordinate `[128, num_classes*4]`
				- 计算 smooth l1 loss 时的 
		'''	
		proposals, gt_bboxes, gt_labels = inputs

		# 计算IoU
		iou = pairwise_iou(proposals, gt_bboxes)
		proposal_max_iou = tf.reduce_max(iou, axis=1)
		proposal_max_iou_indices = tf.argmax(iou, axis=1)

		# labels = tf.zeros((proposals.shape[0],), dtype=tf.int32)
		# 创建　roi_labels，符合特定IoU的gt_labels
		labels = tf.gather(gt_labels, proposal_max_iou_indices)

		# 获取 negative 和 positive
		positive_indices = tf.where(proposal_max_iou >= self._positive_iou_threshold)[:, 0]
		# negative_indices = tf.where(tf.logical_and(proposal_max_iou < self.positive_iou_threshold,
		# 										proposal_max_iou >= self.negative_iou_threshold))[:, 0]
		negative_indices = tf.where(proposal_max_iou < self._positive_iou_threshold)[:, 0]


		# 因为 positive 和 negative 的数量可能过多，所以需要筛选出特定数量的 positive 和 negative
		if tf.size(positive_indices) > self._max_positive_samples:
			positive_indices = tf.random_shuffle(positive_indices)[ :self._max_positive_samples]

		if tf.size(negative_indices) > (self._num_samples - tf.size(positive_indices)):
			negative_incides = tf.random_shuffle(negative_indices)[ :(self._num_samples - tf.size(positive_indices))]
		elif tf.size(negative_indices) == (self._num_samples - tf.size(positive_indices)):
			pass
		else:
			pass

		
		tf.logging.info('rpn_target_proposals_createor generates %d positive and %d negative.' % (tf.size(positive_indices), tf.size(negative_indices)))

		# 因为 `positive_indices` 和 `negative_indices` 都是二维数组，数组的第 0 个维度代表一个坐标，第 1 个维度代表具体的坐标
		# 例如，positive_indices = [[1, 2],
		# 							[3, 4],
		# 							5, 6]]
		# 
		#	   positive_indices = [[7, 8],
		# 							[9, 10],
		# 							[11, 12]]
		#  现在，为了得到总的 labels 坐标，即
		#      proposals = [[1, 2],
		# 					[3, 4],
		# 					[5, 6],
		# 					[7, 8],
		# 					[9, 10],
		# 					[11, 12]]
		# 
		#  其实，就是在这两个数组的第 0 维进行拼接
		final_roi_indices = tf.concat([positive_indices, negative_indices], axis=0)
		final_roi_proposals = tf.gather(proposals, final_roi_indices)
		final_roi_labels = tf.gather(labels, final_roi_indices)

		final_roi_labels = tf.scatter_update(tf.Variable(final_roi_labels), 
											tf.range(tf.size(negative_indices), 
													tf.size(final_roi_indices), dtype=tf.int32), 0)
		# final_roi_labels = tf.where(negative_indices, tf.zeros_like(final_roi_labels), final_roi_labels)

		# w_in 只有正例才会设置，其他均为0
		w_in = tf.zeros((tf.size(final_roi_indices), self._num_classes, 4), dtype=tf.float32)
		
		# 这部分不理解
		if tf.size(positive_indices) > 0:
			w_in = w_in.numpy()
			for index, positive_indices in enumerate(positive_indices.numpy()):
				w_in[index, labels[index]] = 1
		
		'''
			loss 函数为：　L({Pi}, {ti}) = (1/Ncls) * \sum{Lcls(pi, pi*)} 
							            + λ * (1/Nreg) * \sum{pi* * smooth_l1(ti, ti*)}
			
			其实，后一部分的　loss function 实际上为：　λ　* (1/Nreg) * \sum{w_out * smooth_l1(w_in * (ti, ti*))} 
			
		Lreg 其实就是　smooth l1 loss，在具体计算时：
			* 先计算　w_in * (ti - ti*) 
			* 在计算　w_out * smooth_l1(w_in * (ti - ti*))
			* 最后相加除以　num 总数
		
		w_in 在这里面的含义是只计算前景的回归，所以它的定义就是除了前景为 [1, 1, 1, 1]，其余的都是 [0， 0， 0， 0]，而 w_out 是为了在函数中
		加入前景和背景的权重，因为有的时候前景和背景的数量相差悬殊，但是论文中用的是 1:1 的数量，所以对应代码是:
			w_out = np.ones((1, 4)) * 1.0 / num_examples
		相当于前景和背景的 w_out 都是 [1/N_reg, 1/N_reg, 1/N_reg, 1/N_reg]
		'''
		w_in = tf.reshape(w_in, [-1, self._num_classes * 4])

		final_target_bboxes = tf.zeros((tf.size(final_roi_indices), self._num_classes, 4), dtype=tf.float32)
		if tf.size(positive_indices) > 0:
			target_bboxes = encode_bboxes(tf.gather(final_roi_proposals, tf.range(tf.size(positive_indices))),
                                                        tf.gather(gt_bboxes, tf.gather(proposal_max_iou_indices, positive_indices)))
			final_target_bboxes = final_target_bboxes.numpy()
			target_bboxes = target_bboxes.numpy()
			for index, positive_index in enumerate(positive_indices):
				final_target_bboxes[index, labels[index]] = target_bboxes[index]

		final_target_bboxes = tf.reshape(final_target_bboxes, [-1, self._num_classes * 4])

        # 这个好像没啥用
		w_out = tf.ones_like(w_in, dtype=tf.float32)
		return tf.stop_gradient(final_roi_proposals), \
				tf.stop_gradient(final_roi_labels), \
				tf.stop_gradient(final_target_bboxes), \
                tf.stop_gradient(w_in),\
				tf.stop_gradient(w_out)
# import tensorflow as tf

# a = tf.Variable([1, 2, 3, 4], dtype=tf.int32)
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print(tf.size(a).numpy())