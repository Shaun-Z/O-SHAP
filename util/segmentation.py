import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.segmentation import quickshift, slic, watershed
from skimage import segmentation, graph
from skimage.measure import label
from scipy.ndimage import distance_transform_edt, generic_filter
import torch
import cv2

class basic_segment:
    def __init__(self, img):
        '''
        img: np.array, shape=(H, W)
        '''
        H = img.shape[-2]
        W = img.shape[-1]
        self.W = W
        self.H = H
        self.factor = H//28 # 224/28=8
        # self.img = img

        Feature_0 = np.zeros((H,W),dtype = int)

        Feature_1 = np.zeros((H,W),dtype = int)
        Feature_1[:,:self.factor*8], Feature_1[:,self.factor*8:W-self.factor*8], Feature_1[:,W-self.factor*8:]= 0,1,2
        
        Feature_2 = np.zeros((H,W),dtype = int)
        H1, H2, H3= self.factor*8, self.factor*12, self.factor*8
        W1, W2, W3= self.factor*8, self.factor*12, self.factor*8
        Feature_2[:H1,:W1], Feature_2[H1:H1+H2,:W1], Feature_2[H1+H2:H1+H2+H3,:W1] = 0, 1, 2
        Feature_2[:H1,W1:W1+W2], Feature_2[H1:H1+H2,W1:W1+W2], Feature_2[H1+H2:H1+H2+H3,W1:W1+W2] = 3, 4, 5 
        Feature_2[:H1,W1+W2:W1+W2+W3], Feature_2[H1:H1+H2,W1+W2:W1+W2+W3], Feature_2[H1+H2:H1+H2+H3,W1+W2:W1+W2+W3] = 6, 7, 8

        Feature_3 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, self.factor*4):
            for j in range(0, W, self.factor*4):
                Feature_3[i:i+self.factor*4,j:j+self.factor*4] = num
                num += 1

        Feature_4 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, self.factor*2):
            for j in range(0, W, self.factor*2):
                Feature_4[i:i+self.factor*2,j:j+self.factor*2] = num
                num += 1

        Feature_5 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, self.factor*1):
            for j in range(0, W, self.factor*1):
                Feature_5[i:i+self.factor*1,j:j+self.factor*1] = num
                num += 1

        self.features_list = [Feature_0, Feature_1, Feature_2, Feature_3, Feature_4, Feature_5]

    def get_mask(self, feature_ID=5):
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID, savename=None):
        feature = self.get_mask(feature_ID=feature_ID)
        # print(feature)
        # Display heatmap of basic_seg
        plt.figure(figsize=(18,18))
        plt.imshow(feature, cmap='cool', interpolation='nearest')
        for i in range(self.W):
            for j in range(self.H):
                if i > 0 and feature[i, j] != feature[i-1, j]:
                    plt.plot([j-0.5, j+0.5], [i-0.5, i-0.5], color='black', linewidth=1)
                if j > 0 and feature[i, j] != feature[i, j-1]:
                    plt.plot([j-0.5, j-0.5], [i-0.5, i+0.5], color='black', linewidth=1)
                if i < self.W-1 and feature[i, j] != feature[i+1, j]:
                    plt.plot([j-0.5, j+0.5], [i+0.5, i+0.5], color='black', linewidth=1)
                if j < self.H-1 and feature[i, j] != feature[i, j+1]:
                    plt.plot([j+0.5, j+0.5], [i-0.5, i+0.5], color='black', linewidth=1)

        plt.xticks([0, self.W-1], fontsize=35)
        plt.yticks([0, self.H-1], fontsize=35)
        
        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()

    # def __call__(self):
    #     return self.seg

class hierarchical_segment:
    def __init__(self, img, n_segments=10):
        H = img.shape[-2]
        W = img.shape[-1]
        self.W = W
        self.H = H
        self.img = img

        img = self.__preprocess_image(img)
        # 初始过分割
        segments = self.initial_segmentation(img, n_segments)
        # 层次化区域合并
        self.features_list = self.hierarchical_segmentation(img, segments, num_levels=4)
        self.features_list.append(np.zeros((H,W),dtype = int))
        self.features_list.reverse()

    
    def __preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # 转换图像维度
        img = np.transpose(img, (1, 2, 0))
        # 如果图像是浮点型，确保在0-1之间
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 1)
        else:
            img = img / 255.0
        return img
    
    def initial_segmentation(self, img, n_segments=196):
        # 初始过分割，使用 SLIC 超像素
        segments = segmentation.slic(img, n_segments=n_segments, compactness=10, start_label=0)
        return segments
    
    def __build_rag(self, img, segments):
        segments = segments.astype(int)
        # 构建区域邻接图（RAG）
        rag = graph.rag_mean_color(img, segments, mode='similarity')
        return rag
    
    def hierarchical_segmentation(self, img, segments, num_levels=4):
        # 记录每一层的分割结果
        segments_list = [segments]
        current_segments = segments.copy()

        for level in range(1, num_levels):
            if np.unique(current_segments).shape[0] == 1:
                pass
            else:
                # 在每次迭代中重新构建 RAG
                rag = self.__build_rag(img, current_segments)

                # 计算合并阈值，阈值逐层增大，导致合并的区域增多
                thresh = 15 * level * np.percentile([data['weight'] for u, v, data in rag.edges(data=True)], 98)

                # 合并相似区域
                current_segments = graph.merge_hierarchical(
                                    current_segments, rag, thresh=thresh, rag_copy=False,
                                    in_place_merge=False, merge_func=self.__merge_mean_color,
                                    weight_func=self.__weight_mean_color)

            segments_list.append(current_segments.copy())

        return segments_list

    def __merge_mean_color(self, graph_, src, dst):
        # 定义合并后的区域属性更新方式
        graph_.nodes[dst]['total color'] += graph_.nodes[src]['total color']
        graph_.nodes[dst]['pixel count'] += graph_.nodes[src]['pixel count']
        graph_.nodes[dst]['mean color'] = (graph_.nodes[dst]['total color'] /
                                        graph_.nodes[dst]['pixel count'])

    def __weight_mean_color(self, graph_, src, dst, n):
        # 定义区域之间的权重计算方式
        diff = graph_.nodes[dst]['mean color']*255 - graph_.nodes[n]['mean color']*255
        diff = np.linalg.norm(diff)
        return {'weight': diff}
    
    def __display_segments(self, img, segments, title):
        # 绘制分割边缘
        boundary = segmentation.mark_boundaries(img, segments)
        plt.figure(figsize=(8, 8))
        plt.imshow(boundary)
        plt.title(title)
        plt.axis('off')

    def get_mask(self, feature_ID=5):
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID, savename=None):
        feature = self.get_mask(feature_ID=feature_ID)

        self.__display_segments(np.transpose(self.img.detach().cpu().numpy(), (1, 2, 0)), feature, f'Hierarchical Segmentation - Level {feature_ID}')

        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()

class hierarchical_segment_V2:
    def __init__(self, img, n_segments=10):
        H = img.shape[-2]
        W = img.shape[-1]
        self.W = W
        self.H = H
        self.img = img

        img = self.__preprocess_image(img)
        # 初始过分割
        segments = self.initial_segmentation(img, n_segments)
        # 层次化区域合并
        self.features_list = self.hierarchical_segmentation(img, segments, num_levels=4)
        self.features_list.append(np.zeros((H, W), dtype=int))
        self.features_list.reverse()

    def __preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # 转换图像维度
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # (C, H, W) 转换为 (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        # 如果图像是浮点型，确保在0-1之间
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, 0, 1)
        else:
            img = img.astype(np.float64) / 255.0  # 转换为浮点型
        return img

    def initial_segmentation(self, img, n_segments=196):
        """
        初始超像素分割，并根据背景颜色进一步优化。
        """
        # 使用 SLIC 进行初始超像素分割
        segments = segmentation.slic(img, n_segments=n_segments, compactness=10, start_label=0)

        # 提取背景区域的颜色统计
        background_mask = self.__detect_background(img)
        background_color = img[background_mask].mean(axis=0)

        # 优化背景分割：将颜色接近背景的区域标记为背景
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            mean_color = img[mask].mean(axis=0)
            if np.linalg.norm(mean_color - background_color) < 0.4:  # 阈值可调整
                segments[mask] = -1  # 将背景标记为 -1

        # 重标号，确保分割区域连续
        segments = self.__relabel_segments(segments)
        return segments

    def __detect_background(self, img):
        """
        检测背景区域：假设图像四周为背景。
        """
        h, w, _ = img.shape
        margin = 10  # 检测边界的宽度
        mask = np.zeros((h, w), dtype=bool)
        mask[:margin, :] = True  # 上边
        mask[-margin:, :] = True  # 下边
        mask[:, :margin] = True  # 左边
        mask[:, -margin:] = True  # 右边
        return mask
    
    def __relabel_segments(self, segments):
        """
        重标号分割区域，确保标签连续。
        """
        unique_labels, relabeled_segments = np.unique(segments, return_inverse=True)
        return relabeled_segments.reshape(segments.shape)

    def __build_rag(self, img, segments):
        segments = segments.astype(int)
        # 构建区域邻接图（RAG）
        rag = graph.rag_mean_color(img, segments, mode='similarity')
        return rag

    def hierarchical_segmentation(self, img, segments, num_levels=4, background_thresh=0.02):
        """
        层次分割逻辑，背景在第一次合并时单独处理。
        """
        # 记录每一层的分割结果
        segments_list = [segments]
        current_segments = segments.copy()

        # 标记背景区域
        background_segments = set()
        unique_segments = np.unique(current_segments)

        # 检测背景区域：颜色方差低于背景阈值
        for seg_id in unique_segments:
            mask = current_segments == seg_id
            segment_pixels = img[mask]
            if np.var(segment_pixels, axis=0).mean() < background_thresh:
                background_segments.add(seg_id)

        for level in range(1, num_levels):
            if np.unique(current_segments).shape[0] == 1:
                break  # 如果所有区域已合并为一个，则停止

            # 构建 RAG
            rag = self.__build_rag(img, current_segments)

            if level == 1:
                # 第一次合并时，优先合并背景区域
                def weight_func(graph_, src, dst, n):
                    """
                    自定义权重函数：只允许背景区域之间互相合并
                    """
                    if src in background_segments and dst in background_segments:
                        diff = graph_.nodes[dst]['mean color'] * 255 - graph_.nodes[n]['mean color'] * 255
                        return {'weight': np.linalg.norm(diff)}
                    return {'weight': np.inf}  # 禁止非背景区域合并

                current_segments = graph.merge_hierarchical(
                    current_segments, rag, thresh=np.inf, rag_copy=False,
                    in_place_merge=False, merge_func=self.__merge_mean_color,
                    weight_func=weight_func
                )
            else:
                # 后续合并按正常逻辑进行
                thresh = 15 * level * np.percentile(
                    [data['weight'] for u, v, data in rag.edges(data=True)], 98
                )

                current_segments = graph.merge_hierarchical(
                    current_segments, rag, thresh=thresh, rag_copy=False,
                    in_place_merge=False, merge_func=self.__merge_mean_color,
                    weight_func=self.__weight_mean_color
                )

            segments_list.append(current_segments.copy())

        # 如果生成的层数不足 num_levels，则填充
        while len(segments_list) < num_levels:
            segments_list.append(segments_list[-1].copy())  # 复制最后一层填充

        return segments_list

    def __merge_mean_color(self, graph_, src, dst):
        # 定义合并后的区域属性更新方式
        graph_.nodes[dst]['total color'] += graph_.nodes[src]['total color']
        graph_.nodes[dst]['pixel count'] += graph_.nodes[src]['pixel count']
        graph_.nodes[dst]['mean color'] = (
            graph_.nodes[dst]['total color'] / graph_.nodes[dst]['pixel count']
        )

    def __weight_mean_color(self, graph_, src, dst, n):
        # 定义区域之间的权重计算方式
        diff = graph_.nodes[dst]['mean color'] * 255 - graph_.nodes[n]['mean color'] * 255
        diff = np.linalg.norm(diff)
        return {'weight': diff}

    def __display_segments(self, img, segments, title):
        # 确保图像是 numpy 数组
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # 检查图像是否为 (H, W, C)，如果不是则转换
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # (C, H, W) 转换为 (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        # 检查转换后的图像维度是否正确
        if len(img.shape) != 3 or img.shape[2] not in [1, 3]:
            raise ValueError(f"Image shape must be (H, W, C), but got {img.shape}")

        # 确保图像是浮点类型
        if img.dtype not in [np.float32, np.float64]:
            img = img.astype(np.float64)

        # 确保分割标签维度为 (H, W)
        if len(segments.shape) != 2:
            raise ValueError(f"Segments shape must be (H, W), but got {segments.shape}")

        # 绘制分割边界
        boundary = segmentation.mark_boundaries(img, segments)
        plt.figure(figsize=(8, 8))
        plt.imshow(boundary)
        plt.title(title)
        plt.axis('off')

    def get_mask(self, feature_ID=5):
        if feature_ID >= len(self.features_list):
            raise IndexError(f"Requested feature ID {feature_ID} is out of range. "
                             f"Available range: 0 to {len(self.features_list) - 1}.")
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID, savename=None):
        feature = self.get_mask(feature_ID=feature_ID)

        self.__display_segments(
            self.img,
            feature,
            f'Hierarchical Segmentation - Level {feature_ID}'
        )

        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()

class hierarchical_segment_edge:
    def __init__(self, img):
        H = img.shape[-2]
        W = img.shape[-1]
        self.W = W
        self.H = H
        self.img = img

        img = self.__preprocess_image(img)

        # 初始过分割，改为基于 Canny 的分割
        segments = self.initial_segmentation(img)
        # 层次化区域合并
        self.features_list = self.hierarchical_segmentation(img, segments, num_levels=4)
        self.features_list.append(np.zeros((self.H, self.W), dtype=int))
        self.features_list.reverse()

    def __preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # 转换图像维度
        img = np.transpose(img, (1, 2, 0))
        # 如果图像是浮点型，确保在0-1之间
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 1)
        else:
            img = img / 255.0
        return img

    # 定义过滤函数：填充 0（边界像素）为邻域内最多的值
    def __fill_boundary(self, pixel_values):
        center_pixel = pixel_values[len(pixel_values) // 2]
        if center_pixel == 0:  # 如果是边界像素
            valid_values = pixel_values[pixel_values > 0].astype(np.int64)  # 确保是整数
            if len(valid_values) > 0:
                return np.bincount(valid_values).argmax()  # 返回出现次数最多的值
            else:
                return 0  # 如果没有有效值，则保持为 0
        return center_pixel

    def initial_segmentation(self, img):
        """
        使用 Canny 边缘检测和连通域分析生成初始分割。
        """
        # 转换为灰度图
        gray = (img * 255).astype(np.uint8)
        if gray.shape[-1] == 3:  # RGB 转灰度
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        # 高斯模糊，减少噪声
        gray = cv2.GaussianBlur(gray, (5, 5), 1)

        # Canny 边缘检测
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # 膨胀边缘，使其闭合
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭操作

        # 3. 填充区域
        filled_mask = cv2.dilate(closed_edges, kernel, iterations=1)  # 膨胀填充边界
        filled_mask = cv2.threshold(filled_mask, 0, 255, cv2.THRESH_BINARY)[1]

        thin_edges = skeletonize(filled_mask > 0)

        # 假设 thin_edges 是边缘图
        distance = distance_transform_edt(~thin_edges)  # 计算距离变换
        markers = label(distance > 5)  # 根据距离生成标记
        segmented_image = watershed(-distance, markers, mask=~thin_edges)

        segmented_image_no_boundary = generic_filter(segmented_image, self.__fill_boundary, size=3)

        return segmented_image_no_boundary

    def __build_rag(self, img, segments):
        segments = segments.astype(int)
        # 构建区域邻接图（RAG）
        rag = graph.rag_mean_color(img, segments, mode='similarity')
        return rag

    def hierarchical_segmentation(self, img, segments, num_levels=4):
        # 记录每一层的分割结果
        segments_list = [segments]
        current_segments = segments.copy()

        for level in range(1, num_levels):
            if np.unique(current_segments).shape[0] == 1:
                pass
            else:
                # 在每次迭代中重新构建 RAG
                rag = self.__build_rag(img, current_segments)

                # 计算合并阈值，阈值逐层增大，导致合并的区域增多
                thresh = 15 * level * np.percentile(
                    [data['weight'] for u, v, data in rag.edges(data=True)], 98
                )

                # 合并相似区域
                current_segments = graph.merge_hierarchical(
                    current_segments,
                    rag,
                    thresh=thresh,
                    rag_copy=False,
                    in_place_merge=False,
                    merge_func=self.__merge_mean_color,
                    weight_func=self.__weight_mean_color,
                )

            segments_list.append(current_segments.copy())

        return segments_list

    def __merge_mean_color(self, graph_, src, dst):
        # 定义合并后的区域属性更新方式
        graph_.nodes[dst]["total color"] += graph_.nodes[src]["total color"]
        graph_.nodes[dst]["pixel count"] += graph_.nodes[src]["pixel count"]
        graph_.nodes[dst]["mean color"] = (
            graph_.nodes[dst]["total color"] / graph_.nodes[dst]["pixel count"]
        )

    def __weight_mean_color(self, graph_, src, dst, n):
        # 定义区域之间的权重计算方式
        diff = graph_.nodes[dst]["mean color"] * 255 - graph_.nodes[n]["mean color"] * 255
        diff = np.linalg.norm(diff)
        return {"weight": diff}

    def __display_segments(self, img, segments, title):
        # 绘制分割边缘
        boundary = segmentation.mark_boundaries(img, segments)
        # plt.figure(figsize=(8, 8))
        plt.imshow(boundary)
        plt.title(title)
        plt.axis("off")

    def get_mask(self, feature_ID=5):
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID, savename=None):
        feature = self.get_mask(feature_ID=feature_ID)

        self.__display_segments(np.transpose(self.img.detach().cpu().numpy(), (1, 2, 0)), feature, f'Hierarchical Segmentation - Level {feature_ID}')

        if savename is not None:
            plt.savefig(savename)
        else:
            plt.show()