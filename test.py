import numpy as np
from PIL import Image
from colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from dct import split_into_blocks, merge_blocks, create_dct_matrix, dct_2d, idct_2d
from quantization import scale_quantization_table, QT_LUMINANCE, QT_CHROMINANCE, quantize, dequantize
from zigzag import zigzag_8x8, inverse_zigzag_8x8

with open("Тестовые данные/RAW_color.raw", 'rb') as f:
    header = f.read(7)
    magic = header[0]
    img_type = header[1]
    colorspace_id = header[2]
    width = header[3] | (header[4] << 8)
    height = header[5] | (header[6] << 8)
    img_data = f.read()

quality = 90
qt_lum = scale_quantization_table(QT_LUMINANCE, quality)
qt_chrom = scale_quantization_table(QT_CHROMINANCE, quality)

img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
ycbcr = rgb_to_ycbcr(img)
channels1 = [ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]]
blocks = []
rows_blocks = []
cols_blocks = []
coeffs_ch = []
q_tables = [qt_lum, qt_chrom, qt_chrom]

for ch, qt in zip(channels1, q_tables):
    blocks_, rows_blocks_, cols_blocks_, orig = split_into_blocks(ch, 8)
    coeffs = []
    for block in blocks_:
        coeff = dct_2d(block.astype(np.float64) - 128)
        quant = quantize(coeff, qt)
        zz = zigzag_8x8(quant)
        coeffs.append(zz)
    coeffs_ch.append(coeffs)
    rows_blocks.append(rows_blocks_)
    cols_blocks.append(cols_blocks_)
#######

channels = []
qtab = [qt_lum, qt_chrom, qt_chrom]
for i in range(3):
    blocks_ch = []
    for cf in coeffs_ch[i]:
        quant = inverse_zigzag_8x8(cf)
        deq = dequantize(quant.astype(np.float64), qtab[i])
        block = idct_2d(deq.astype(np.float64)) + 128
        block = np.clip(block.astype(np.float64), 0, 255)
        blocks_ch.append(block)
    channel = merge_blocks(blocks_ch, rows_blocks[i], cols_blocks[i], 8)
    channels.append(channel)
y = channels[0][:height, :width]
cb = channels[1][:height, :width]
cr = channels[2][:height, :width]
ycbcr = np.stack((y, cb, cr), axis=-1)
rgb = ycbcr_to_rgb(ycbcr)
pil_img = Image.fromarray(rgb, 'RGB')
pil_img.save("decompressed.jpg")