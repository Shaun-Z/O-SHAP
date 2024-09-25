import numpy as np

def calculate_ebpg(xai_result, mask, class_index):
    """
    xai_result: shape (N, 224, 224, 3), N个解释图
    mask: shape (224, 224)，目标区域的mask，类别对应为1-20
    class_index: 指定类别的索引（1-20）
    返回：平均EBPG
    """
    energy_in_mask = []
    
    # 只计算mask中等于class_index的区域
    target_mask = (mask == class_index)
    
    # 对每个XAI结果，计算其能量集中于mask的比例
    for xai_map in xai_result:
        # 计算XAI结果的L1范数能量
        total_energy = np.sum(np.abs(xai_map))
        
        # 计算mask区域内的能量
        mask_energy = np.sum(np.abs(xai_map[target_mask]))
        
        # 计算EBPG (能量集中于mask的比例)
        energy_in_mask.append(mask_energy / total_energy if total_energy > 0 else 0)
    
    return np.mean(energy_in_mask)

# # 调用函数
# ebpg_score = calculate_ebpg(xai_result, mask)
# print(f"EBPG Score: {ebpg_score}")

def calculate_miou(xai_result, mask, class_index, threshold=0.8):
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
        
        # 获取前20%像素的阈值
        sorted_pixels = np.sort(saliency_map.flatten())
        top_threshold = sorted_pixels[int(len(sorted_pixels) * (1 - threshold))]
        
        # 生成前20%最重要像素的二值化掩码
        pred_mask = saliency_map >= top_threshold
        
        # 计算mIoU
        intersection = np.sum((pred_mask & target_mask))
        union = np.sum(pred_mask | target_mask)
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    
    return np.mean(iou_scores)

# # 调用函数
# miou_score = calculate_miou(xai_result, mask)
# print(f"mIoU Score: {miou_score}")

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
    num_mask_pixels = np.sum(target_mask)
    
    for xai_map in xai_result:
        # 将XAI解释图降到2D
        saliency_map = np.mean(xai_map, axis=-1)
        
        # 选取前N个重要像素
        sorted_pixels = np.argsort(saliency_map.flatten())[::-1]
        top_n_pixels = sorted_pixels[:num_mask_pixels]
        
        # 创建预测的bbox掩码
        pred_mask = np.zeros_like(saliency_map, dtype=bool)
        pred_mask[np.unravel_index(top_n_pixels, saliency_map.shape)] = True
        
        # 计算Bounding Box得分
        intersection = np.sum((pred_mask & target_mask))
        bbox_scores.append(intersection / num_mask_pixels if num_mask_pixels > 0 else 0)
    
    return np.mean(bbox_scores)

# 调用函数
# bbox_score = calculate_bbox(xai_result, mask)
# print(f"Bbox Score: {bbox_score}")