import numpy as np

def calculate_ebpg(xai_result, mask, class_index):
    """
    xai_result: shape (N, 224, 224, 3), N个解释图
    mask: shape (224, 224)，目标区域的mask，类别对应为1-20
    class_index: 指定类别的索引（1-20）
    返回：平均EBPG
    """
    energy_true_rate = []
    
    # 只计算mask中等于class_index的区域
    target_mask = (mask == class_index)
    
    # 对每个XAI结果，计算其能量集中于mask的比例
    for xai_map in xai_result:
        # 计算XAI结果的L1范数能量
        total_energy = np.sum(np.abs(xai_map))
        total_energy_p = np.sum(xai_map[xai_map >= 0])
        total_energy_n = -np.sum(xai_map[xai_map < 0])
        
        # 计算mask区域内的能量
        xai_map_mask = xai_map[target_mask]
        # 计算非mask区域内的能量
        xai_map_unmask = xai_map[~target_mask]
        
        # True Positive and False Negative
        energy_TP = np.sum(xai_map_mask[xai_map_mask >= 0])
        energy_FN = -np.sum(xai_map_mask[xai_map_mask < 0])
        # False Positive and True Negative
        energy_FP = np.sum(xai_map_unmask[xai_map_unmask >= 0])
        energy_TN = -np.sum(xai_map_unmask[xai_map_unmask < 0])
        
        # 计算EBPG (能量集中于mask的比例)
        energy_true_rate.append((energy_TP+energy_TN) / total_energy)
    
    return np.mean(energy_true_rate)

# # 调用函数
# ebpg_score = calculate_ebpg(xai_result, mask)
# print(f"EBPG Score: {ebpg_score}")

def calculate_miou(xai_result, mask, class_index, threshold=0.001):
    """
    xai_result: shape (N, 224, 224, 3), N个解释图
    mask: shape (224, 224)，目标区域的mask，类别对应为1-20
    class_index: 指定类别的索引（1-20）
    threshold: 前20%的像素选取阈值
    返回：平均mIoU
    """
    iou_scores = []
    
    # 只计算mask中等于class_index的区域
    target_mask = (mask == class_index)
    
    for xai_map in xai_result:
        # 将XAI解释图降到2D (只使用灰度)
        saliency_map = np.mean(xai_map, axis=-1)
        
        # 将saliency_map展平为一维数组
        flat_saliency = saliency_map.flatten()
        
        # 计算前20%像素的数量
        num_top_pixels = int(len(flat_saliency) * threshold)
        
        # 获取前20%像素值的阈值
        top_pixels_threshold = np.partition(flat_saliency, -num_top_pixels)[-num_top_pixels]
        
        # 生成前20%最重要像素的二值化掩码
        pred_mask = saliency_map >= top_pixels_threshold
        
        # 计算交集和并集的像素数量
        intersection = np.sum((pred_mask & target_mask))
        union = np.sum((pred_mask | target_mask))
        
        # 计算mIoU (像素数量的比值)
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    
    return np.mean(iou_scores)

def calculate_bbox(xai_result, mask, class_index):
    """
    xai_result: shape (N, 224, 224, 3), N个解释图
    mask: shape (224, 224)，目标区域的mask，类别对应为1-20
    class_index: 指定类别的索引（1-20）
    返回：平均Bbox得分
    """
    bbox_scores = []
    
    # 只计算mask中等于class_index的区域
    target_mask = (mask == class_index)
    
    # 计算mask中目标类别的像素数
    num_mask_pixels = np.sum(target_mask) // 8
    
    for xai_map in xai_result:
        # 将XAI解释图降到2D
        saliency_map = np.mean(xai_map, axis=-1)
        
        # 获取前N个最重要的像素索引
        sorted_indices = np.argsort(saliency_map.flatten())[::-1]
        top_n_pixels = sorted_indices[:num_mask_pixels]
        
        # 创建预测的bbox掩码（仅保留前N个像素）
        pred_mask = np.zeros_like(saliency_map, dtype=bool)
        pred_mask[np.unravel_index(top_n_pixels, saliency_map.shape)] = True
        
        # 计算交集和目标区域的重合率
        intersection = np.sum((pred_mask & target_mask))
        bbox_scores.append(intersection / num_mask_pixels)
    
    return np.mean(bbox_scores)
