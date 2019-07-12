import tensorflow as tf

'''
		rpn 中的 loss 函数为：
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
		'''		

def rpn_losses_creator(num_anchors, rpn_scores, rpn_bbox_coordinates,
                        rpn_labels, rpn_target_anchors, rpn_w_in, rpn_w_out):

    # 因为 `rpn_scores` 的 shape 为 `[H * W * 18]`
    # `[H * W * 18]` ---> `[H * W, 2, 9]`
    rpn_scores = tf.reshape(rpn_scores, [-1, 2, num_anchors])

    # `[H * W, 2, 9]` ---> `[H * W, 9, 2]`
    rpn_scores = tf.transpose(rpn_scores, [0, 2, 1])

    # `[H * W, 9, 2]` ---> `[H * W * 9, 2]`
    rpn_scores = tf.reshape(rpn_scores, [-1, 2])

    rpn_target_anchors_indices = tf.where(rpn_labels >= 0)[:, 0]
    
    # 个人感觉这样写有毛病，不应该直接是 `rpn_pi_star` 吗
    # rpn_target_labels = tf.gather(rpn_labels, rpn_target_anchors_indices)
  
    rpn_scores = tf.gather(rpn_scores, rpn_target_anchors_indices)

    rpn_labels = tf.cast(tf.gather(rpn_labels, rpn_target_anchors_indices), dtype=tf.int32)

    # 个人感觉这样写才符合原论文
    rpn_cls_loss = tf.losses.sparse_softmax_cross_entropy(rpn_labels, logits=rpn_scores)

    rpn_reg_loss = smooth_l1_loss(rpn_bbox_coordinates, rpn_target_anchors, rpn_w_in, rpn_w_out)
    
    return rpn_cls_loss, rpn_reg_loss


def roi_losses_creator(roi_scores, roi_coordinates, 
                      final_roi_labels, final_roi_target_bboxes,
                      roi_w_in, roi_w_out):
    roi_cls_loss = tf.losses.sparse_softmax_cross_entropy(logits=roi_scores,
                                                          labels=final_roi_labels)

    roi_reg_loss = smooth_l1_loss(roi_coordinates, final_roi_target_bboxes, 
                                  roi_w_in, roi_w_out)
    return roi_cls_loss, roi_reg_loss


# FIXME：这里有问题
def smooth_l1_loss(pred_bboxes,
                   target_anchors,
                   w_in,
                   w_out,
                   sigma=1.0):
    '''
    由于在实际过程中，`Ncls` 和 `Nreg` 差距过大，用参数 `λ` 平衡二者（如 `Ncls = 256`，`Nreg = 2400` 时，
	设置 `λ = Nreg / Ncls ≈ 10`，使总的网络 Loss 计算过程中能够均匀考虑两种 Loss。
		
     λ * (1/Nreg) * \sum{pi* * Lreg(ti, ti*)}

    Lreg(ti, ti*) = \sum {smooth_L1(ti - ti*)}
			
						  |---- 0.5x^2      if |x| < 1 
			smooth_L1(x) = |
						  |---- |x| - 0.5   otherwise

    λ　* (1/Nreg) * \sum{w_out * smooth_l1(w_in * (ti, ti*))}
    '''
    sigma = sigma ** 2
    x = w_in * (pred_bboxes - target_anchors)
    abs_x = tf.abs(x)
    #这里sign的值就是1或者-1
    sign = tf.stop_gradient(tf.cast(tf.less(abs_x, 1. / sigma), dtype=tf.float32))
    reg_loss = tf.reduce_mean(
      tf.reduce_sum(
        w_out * (tf.pow(x, 2) * (sigma / 2.) * sign + (abs_x - (0.5 / sigma)) * (1. - sign)),
        axis=1)
    )
    return reg_loss
                