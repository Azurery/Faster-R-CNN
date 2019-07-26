import numpy as np
import tensorflow as tf


# def generate_anchors_base():
	


# def generate_anchors(anchor_step, anchor_sizes, anchor_ratios):
# 	'''
# 	其实，`anchor_step` 指的是当前 feature map 在原图上的投影，简而言之，就是 feature map 上的
# 	一个点相当于原图上的 `anchor_step` 个位置
# 	'''

# 	# `anchor_step / 2` 这样做的目的是：得到的 bbox 的中心坐标，因为这个坐标是相对偏移量，即 `offset`
# 	py = anchor_step / 2.
# 	px = anchor_step / 2.

# 	'''
# 	生成一个 `[num_anchors, 4]` 维的向量，表示一共生成 `num_anchors` 个 anchor，每个 anchor 的坐标为 `(x, y, h, w)`
# 	'''
# 	# anchors = tf.zeros([len(anchor_sizes) * len(anchor_ratios), 4], dtype=tf.float32)
# 	anchors = np.zeros((len(anchor_sizes) * len(anchor_ratios), 4), dtype=np.float32)
# 	for i in range(len(anchor_ratios)):
# 		for j in range(len(anchor_sizes)):
# 			# 不太理解为甚么要乘以根号下，而不是直接相乘
# 			h = anchor_step * anchor_sizes[j] * np.sqrt(anchor_ratios[i])
# 			w = anchor_step * anchor_sizes[j] * np.sqrt(1. / anchor_ratios[i])

# 			index = i * len(anchor_sizes) + j
# 			anchors[index, 0] = py - h / 2.
# 			anchors[index, 1] = px - w / 2.
# 			anchors[index, 2] = py + h / 2.
# 			anchors[index, 3] = px + w / 2.
# 	return anchors


def generate_anchors(feature_map_width, feature_map_height, base_anchor_size, 
					feature_stride, anchor_ratios, anchor_steps):
	'''
	此函数决定了最终 anchors 的长宽，后续 generate_by_anchor_base 函数的作用是确定anchor的中心点
    输入的三个参数都会影响到最终的长宽：
    anchir_ratios 确定了长宽的比例
    base_size 和 anchor_steps 共同决定了 anchor 的具体尺寸，即 base_size * anchor_steps 就是最终 anchors 的尺寸
	'''
	
	base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
	
	# [128, 256, 512] 三个基本尺寸
	# [[  0.   0. 128. 128.]
	#  [  0.   0. 256. 256.]
	#  [  0.   0. 512. 512.]], shape=(3, 4), dtype=float32)
	base_anchors = base_anchor * tf.constant(anchor_steps, shape=(len(anchor_steps), 1), dtype=tf.float32)

	# 将得到一个[3, 1]的矩阵
	# tf.Tensor(
			# [[0.70710677]
			#  [1.        ]
			#  [1.4142135 ]], shape=(3, 1), dtype=float32)
	sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))[:, tf.newaxis]

	# reshape前：
	# <tf.Tensor: id=21, shape=(3, 3), dtype=float32, numpy=
	# 	array([[181.01933, 362.03867, 724.07733],
	# 		   [128.     , 256.     , 512.     ],
	# 		   [ 90.50967, 181.01933, 362.03867]], dtype=float32)>, 

	# <tf.Tensor: id=16, shape=(3, 3), dtype=float32, numpy=
	# 	array([[181.01933, 362.03867, 724.07733],
	# 		   [128.     , 256.     , 512.     ],
	# 		   [ 90.50967, 181.01933, 362.03867]], dtype=float32)>)
	anchors_w = base_anchors[:, 2] / sqrt_ratios
	anchors_h = base_anchors[:, 3] * sqrt_ratios

	# reshape之后,变成[9, 1]矩阵
	# array([[181.01933],
		#    [362.03867],
		#    [724.07733],
		#    [128.     ],
		#    [256.     ],
		#    [512.     ],
		#    [90.50967],
		#    [181.01933],
		#    [362.03867]], dtype=float32)>
	anchors_h = tf.reshape(anchors_h, [-1, 1])
	anchors_w = tf.reshape(anchors_w, [-1, 1])

	# 一张feature map一共有feature_map_width*feature_map_height个点，而由于每个点
	# 又有９个anchors,所以每个点都需要９个相同的坐标
	# 下面用于先生成feature_map_width*feature_map_height个点
	x, y = tf.meshgrid(tf.range(feature_map_width, dtype=tf.float32) * feature_stride,
					tf.range(feature_map_height, dtype=tf.float32) * feature_stride)
	
	# shape为[9, feature_map_width]
	anchors_w, x = tf.meshgrid(anchors_w, x)
	anchors_h, y = tf.meshgrid(anchors_h, y)

	# reshape之前：[feature_map_width*feature_map_height, 9, 2]
	# shape变为[9*feature_map_width*feature_map_height, 2]
	anchors_xy = tf.reshape(tf.stack([x, y], axis=2), [-1, 2])

	anchors_wh = tf.reshape(tf.stack([anchors_w, anchors_h], axis=2), [-1, 2])
	anchors = tf.concat([anchors_xy - 0.5 * anchors_wh, anchors_xy + 0.5 * anchors_wh], axis=1)
	# print(anchors)
	return anchors

def encode_bboxes(src_bboxes, dst_bboxes):
	src_bboxes = tf.cast(src_bboxes, tf.float32)
	gt_bboxes = tf.cast(dst_bboxes, tf.float32)

	w = src_bboxes[:, 2] - src_bboxes[:, 0]
	h = src_bboxes[:, 3] - src_bboxes[:, 1]
	center_x = src_bboxes[..., 0] + 0.5 * w
	center_y = src_bboxes[..., 1] + 0.5 * h

	gt_width = gt_bboxes[..., 2] - gt_bboxes[..., 0] + 1.0
	gt_height = gt_bboxes[..., 3] - gt_bboxes[..., 1] + 1.0
	gt_center_x = gt_bboxes[..., 0] + 0.5 * gt_width
	gt_center_y = gt_bboxes[..., 1] + 0.5 * gt_height

	dx = (gt_center_x - center_x) / w
	dy = (gt_center_y - center_y) / h
	dw = tf.log(gt_width / w)
	dh = tf.log(gt_height / h)

	encoded_bboxes = tf.stack([dx, dy, dw, dh], axis=-1)
	return encoded_bboxes
	


def decode_bboxes(bboxes, rpn_coordinates):
	'''
	用于将预测得到的 bboxes 进行解码，得到 anchors 真正的位置坐标

	转换公式为：
		`\\hat{g}_y = p_h t_y + p_y`
		`\\hat{g}_x = p_w t_x + p_x`
		`\\hat{g}_h = p_h \\exp(t_h)`
		`\\hat{g}_w = p_w \\exp(t_w)`
	'''
	if bboxes.get_shape().as_list()[0] == 0:
		return np.zeros((0, 4), dtype=rpn_coordinates.dtype)
	
    # 其中，`Ah`、`Aw`、`Ay`、`Ax` 分别表示给定 anchor 的坐标
	Ah = bboxes[:, 2] - bboxes[:, 0]
	Aw = bboxes[:, 3] - bboxes[:, 1]
	Ay = bboxes[:, 0] + .5 * Ah
	Ax = bboxes[:, 1] + .5 * Aw

    # 其中，`dy`、`dx`、`dh`、`dw` 是来自预测值
	dy = rpn_coordinates[:, 0]
	dx = rpn_coordinates[:, 1]
	dh = rpn_coordinates[:, 2]
	dw = rpn_coordinates[:, 3]

	# 先进行平移操作
	Gy = Ah * dy + Ay
	Gx = Aw * dx + Ax

    # 进行缩放操作
	Gw = Aw * tf.exp(dw)
	Gh = Ah * tf.exp(dh)


    # 得到边界框的左上角坐标 `(xmin, ymin)` 和右下角坐标 `(xmax, ymax)`
	xmin = Gx - .5 * Gw
	ymin = Gy - .5 * Gh
	xmax = Gx + .5 * Gw
	ymax = Gy + .5 * Gh

	decoded_bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=1)
	return decoded_bboxes

def clip_bboxes(anchors, max_height, max_width, min_edge=None):
	'''
	作用：
		根据边界、最小边长来过滤 anchors
	'''

	# 得到 `anchors` 的第 `1` 个维度，即 `anchors` 的坐标 `(xmin, ymin, xmax, ymax)`
	# `[num_anchors * feature_width * feature_height, 4]`
	bboxes_coord = tf.split(anchors, 4, axis=1)
	
	# 这么做的目的是：使得 roi 的 x 坐标范围在 `[0, max_width]`，y 坐标范围为 `[0, max_height]`
	# 因为有的 anchors 的边界已经超出了图像本身，所以需要将超出图像范围的边界坐标设置为 `0` 或图像宽高
	bboxes_coord[0] = tf.maximum(tf.minimum(bboxes_coord[0], max_width), 0)
	bboxes_coord[1] = tf.maximum(tf.minimum(bboxes_coord[1], max_height), 0)
	bboxes_coord[2] = tf.maximum(tf.minimum(bboxes_coord[2], max_width), 0)
	bboxes_coord[3] = tf.maximum(tf.minimum(bboxes_coord[3], max_height ), 0)

	proposals = tf.concat(bboxes_coord, axis=1)
	return proposals


def filter_overlap_bboxes(anchors, image_shape):
	'''
	作用：
		过滤 anchors，主要是将那别超过图像大小范围的 anchors 都不要
	'''
	filtered_indicies = tf.where(
		tf.logical_and(
			tf.logical_and((anchors[:, 0] >= 0), (anchors[:, 1] >=0)),
			tf.logical_and((anchors[:, 2] <= image_shape[0].numpy()), (anchors[:, 3] <= image_shape[1].numpy()))
		)
	)[:, 0]
	
	filtered_bboxes = tf.gather(anchors, filtered_indicies)
	return filtered_indicies, filtered_bboxes

# FIXME：感觉这么写可读性比较差，虽然形式比较简单
# 最后还是应该改写成遍历的形式
def pairwise_iou(anchors, gt_bboxes):
	'''
	原理：
		https://pro.arcgis.com/en/pro-app/tool-reference/analysis/how-pairwise-intersect-works.htm
	作用：
		主要是用于计算两个 bboxes 集合中对应 bbox 之间的 IoU
	
	其中，`bboxes_a` 和 `bboxes_b ` 表示拥有 `N` 个和 `M` 个 bbox 的集合
	- `bboxes_a` 为 `N * 4` 的 bboxes list
	- `bboxes_b` 为 `M * 4` 的 bboxes list
	'''
	
	def _intersection_area(bboxes):
		xmin, ymin, xmax, ymax = tf.split(bboxes, 4, axis=1)
		area = tf.squeeze((ymax - ymin) * (xmax - xmin), [1])
		return area

	anchors = tf.cast(anchors, tf.float32)
	gt_bboxes = tf.cast(gt_bboxes, tf.float32)
	# print('gt_bboxes:', gt_bboxes)

	# `anchors` 的 shape：`[num_anchors, 4]`
	# 下面这么做的目的：是为了得到 `anchors` 中每一个 anchor 的坐标，以及
	# `gt_bboxes` 中每一个 gt_bbox 的坐标 	
	anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax = tf.split(anchors, 4, axis=1)
	gt_xmin, gt_ymin, gt_xmax, gt_ymax = tf.split(gt_bboxes, 4, axis=1)

	'''
	解释：
		https://www.twblogs.net/a/5c8825dabd9eee35cd6a5129
	
	下面操作的目的：为了计算 `anchors` 和 `gt_bboxes` 这两个 bboxes 中任意一对 bbox的相交面积
					这个相交面积的操作有点儿类似于笛卡尔积

	【注意】`tf.minimum` 和 `tf.maximum` 这两个函数要求 `x` 和 `y` 的 shape 必须相同
	然而下面对 gt_bboxes 进行 transpose 的目的是为了进行了 broadcast，即进行广播操作
	即如果支持广播操作，可以认为 `x` 依次与 `y` 中的每一个数进行 minimum 操作，从而最终得到的 shape 将为：`[M * N]`  
	
	简而言之，就是如果不对 `y` 进行 transpose 操作的话，那么就将是 `x` 与 `y` 的对应元素进行 minimum 操作，而不是将 `x` 中的一个
	元素与 `y` 中的所有元素进行 minimum 操作
	'''
	ymin = tf.maximum(anchor_ymin, tf.transpose(gt_ymin))
	xmin = tf.maximum(anchor_xmin, tf.transpose(gt_xmin))
	ymax = tf.minimum(anchor_ymax, tf.transpose(gt_ymax))
	xmax = tf.minimum(anchor_xmax, tf.transpose(gt_xmax))

	h = tf.maximum(ymax - ymin, 0.)
	w = tf.maximum(xmax - xmin, 0.)
	intersections = h * w

	anchor_areas = _intersection_area(anchors)
	gt_areas = _intersection_area(gt_bboxes)
	# print('gt_areas:', gt_areas)
	
	unions = (tf.expand_dims(anchor_areas, 1) + tf.expand_dims(gt_areas, 0) - intersections)
	iou = tf.where(tf.equal(intersections, 0.0),
					tf.zeros_like(intersections),
					tf.truediv(intersections, unions))
	return iou


		



