import numpy as np
import math
from Main import read_raw_file, get_pixel_data, raw_to_jpg

def downsample_decimate(pixel_data, width, height, channels=3):
    new_width = width // 2
    new_height = height // 2

    if channels == 3:
        img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)
        downsampled = img_array[::2, ::2, :]
    else:
        img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width)
        downsampled = img_array[::2, ::2]

    new_data = downsampled.tobytes()

    print(f"Даунсэмплинг: {width}x{height} to {new_width}x{new_height}")
    print(f"Размер данных: {len(pixel_data):,} to {len(new_data):,} байт")

    return new_data, new_width, new_height

def upsample_pixel(pixel_data, width, height, target_width, target_height, channels=3):
    img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)
    new_data = np.repeat(np.repeat(img_array, target_height // height, axis=0), target_width // width, axis=1)
    return new_data.tobytes()

def linear_interpolation(x1, x2, y1, y2, x):
    if x1 == x2:
        return y1
    t = (x - x1) / (x2 - x1)
    result = y1 + t * (y2 - y1)

    return result

def linear_spline(x_points, y_points, x):
    for i in range(len(x_points) - 1):
        if x_points[i] <= x <= x_points[i + 1]:
            return linear_interpolation(x_points[i], x_points[i + 1],
                                        y_points[i], y_points[i + 1], x)
    if x < x_points[0]:
        return y_points[0]
    else:
        return y_points[-1]

def bilinear_interpolation(x1, x2, y1, y2, z11, z12, z21, z22, x, y):
    x_y1 = linear_interpolation(x1, x2, z11, z21, x)
    x_y2 = linear_interpolation(x1, x2, z12, z22, x)
    result = linear_interpolation(y1, y2, x_y1, x_y2, y)
    return result

def upsample_bilinear(pixel_data, width, height, target_width, target_height, channels=3):
    img_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)
    new_array = np.zeros((target_height, target_width, channels), dtype=np.uint8)

    scale_x = (width - 1) / (target_width - 1)
    scale_y = (height - 1) / (target_height - 1)

    for y in range(target_height):
        for x in range(target_width):
            src_x = x * scale_x
            src_y = y * scale_y
            x1 = int(math.floor(src_x))
            x2 = min(x1 + 1, width - 1)
            y1 = int(math.floor(src_y))
            y2 = min(y1 + 1, height - 1)
            fx = src_x - x1
            fy = src_y - y1

            for c in range(channels):
                z11 = img_array[y1, x1, c]
                z21 = img_array[y1, x2, c]
                z12 = img_array[y2, x1, c]
                z22 = img_array[y2, x2, c]
                x_y1 = z11 * (1 - fx) + z21 * fx
                x_y2 = z12 * (1 - fx) + z22 * fx
                new_array[y, x, c] = int(x_y1 * (1 - fy) + x_y2 * fy)
    return new_array.tobytes()

def save_raw_image(pixel_data, width, height, img_type, colorspace_id, output_path):
    header = bytearray(7)
    header[0] = 0xAB
    header[1] = img_type
    header[2] = colorspace_id
    header[3] = width & 0xFF
    header[4] = (width >> 8) & 0xFF
    header[5] = height & 0xFF
    header[6] = (height >> 8) & 0xFF

    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(pixel_data)

# print("Down/UpSampling")
#
# print("---Даунсемпл по ближайшему соседу")
# img_data = get_pixel_data("Тестовые данные/RAW_color.raw")
# down_dec, ddw, ddh = downsample_decimate(img_data, 512, 512, 3)
# save_raw_image(down_dec, ddw, ddh, 0x03, 0x01, "Данные на проверку/RAW_down_dec_color.raw")
# raw_to_jpg("Данные на проверку/RAW_down_dec_color.raw", "Данные на проверку/down_dec_color.jpeg")
#
# print("---Апсемпл по ближайшему соседу")
# down_data = get_pixel_data("Данные на проверку/RAW_down_dec_color.raw")
# up_pix = upsample_pixel(down_data, 256, 256, 512, 512, 3)
# save_raw_image(up_pix, 512, 512, 0x03, 0x01, "Данные на проверку/RAW_up_pix_color.raw")
# raw_to_jpg("Данные на проверку/RAW_up_pix_color.raw", "Данные на проверку/up_pix_color.jpeg")
#
# print("---Апсемпл с билинейной интерполяцией")
# up_bil = upsample_bilinear(down_data, 256, 256, 512, 512, 3)
# save_raw_image(up_bil, 512, 512, 0x03, 0x01, "Данные на проверку/RAW_up_bil_color.raw")
# raw_to_jpg("Данные на проверку/RAW_up_bil_color.raw", "Данные на проверку/up_bil_color.jpeg")