import numpy as np
from PIL import Image
import torch
import cv2

# Tensor to PIL


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class ImageMeanshiftFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "sp": ("INT", {"default": 20}),
                "sr": ("INT", {"default": 20}),
                "use_cuda": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_meanshift_filter"
    CATEGORY = "Tools"

    def meanshift_denoise_pil_cuda(self, input_image_pil, sp=80, sr=60):
        # 将 PIL 图像转换为 OpenCV 图像格式
        image = cv2.cvtColor(np.array(input_image_pil), cv2.COLOR_RGB2BGR)

        # 将图像转换为 RGBA
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # 创建 GPU 矩阵并上传图像
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image_rgba)

        # 应用 MeanShift 滤波
        filtered_gpu_image = cv2.cuda.meanShiftFiltering(gpu_image, sp, sr)

        # 下载处理后的图像回 CPU
        filtered_image = filtered_gpu_image.download()

        # 将结果图像转换回 BGR，然后转换为 RGB
        filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGRA2RGB)

        # 将 OpenCV 图像转换回 PIL 图像
        output_image_pil = Image.fromarray(filtered_image_rgb)

        return output_image_pil

    def meanshift_denoise_pil_cpu(self, input_image_pil, sp, sr):
        # 将 PIL 图像转换为 OpenCV 图像格式
        image = cv2.cvtColor(np.array(input_image_pil), cv2.COLOR_RGB2BGR)

        # 检测是否有透明度通道
        if image.shape[2] == 4:
            # 分离透明度通道
            alpha_channel = image[:, :, 3].copy()
            image = image[:, :, :3]
        else:
            alpha_channel = None

        # 应用 MeanShift 滤波
        filtered_image = cv2.pyrMeanShiftFiltering(image, sp, sr)

        # 如果原图有透明度通道，将其添加回处理后的图像
        if alpha_channel is not None:
            filtered_image_with_alpha = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2BGRA)
            filtered_image_with_alpha[:, :, 3] = alpha_channel
            filtered_image = filtered_image_with_alpha

        # 将结果图像转换回 RGB
        filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB if alpha_channel is None else cv2.COLOR_BGRA2RGBA)

        # 将 OpenCV 图像转换回 PIL 图像
        output_image_pil = Image.fromarray(filtered_image_rgb)

        return output_image_pil
    
    def apply_meanshift_filter(self, image, sp, sr, use_cuda):
        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)

                # Apply Fliter
                if use_cuda:
                    new_img = self.meanshift_denoise_pil_cuda(pil_image, sp, sr)
                else:
                    new_img = self.meanshift_denoise_pil_cpu(pil_image, sp, sr)

                # Output image
                out_image = (pil2tensor(new_img) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:
            pil_image = None
            img = image
            # PIL Image

            pil_image = tensor2pil(img)

             # Apply Fliter
            if use_cuda:
                new_img = self.meanshift_denoise_pil_cuda(pil_image, sp, sr)
            else:
                new_img = self.meanshift_denoise_pil_cpu(pil_image, sp, sr)
            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, )


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageMeanshiftFilter": ImageMeanshiftFilter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMeanshiftFilter": "Apply Meanshift Filter"
}
