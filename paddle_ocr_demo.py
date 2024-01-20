
import hashlib
import datetime
import os
import fitz  # pip install PyMuPDF==1.19.0
import cv2
from paddleocr import PaddleOCR
import os
import shutil
import requests, io

BASE_DIR = "./"
MODELFILE_PATH = "/data/models/"

# 预先下载好的模型路径
det_model_dir = os.path.join(MODELFILE_PATH, "paddleocr/det")
rec_model_dir = os.path.join(MODELFILE_PATH, "paddleocr/rec")
cls_model_dir = os.path.join(MODELFILE_PATH, "paddleocr/cls")

# 加载OCR引擎
engine = PaddleOCR(enable_mkldnn=True,
                   use_angle_cls=False,
                   det_model_dir=det_model_dir,
                   rec_model_dir=rec_model_dir,
                   cls_model_dir=cls_model_dir)


def pyMuPDF_fitz(pdfPath, imagePath):
    startTime_pdf2img = datetime.datetime.now()  # 开始时间

    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        # 此处若是不做设置，默认图片大小为：792X612, dpi=96
        zoom_x = 1.33333333  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)
        if not os.path.exists(imagePath):  # 判断存放图片的文件夹是否存在
            os.makedirs(imagePath)  # 若图片文件夹不存在就创建
        pix.writePNG(imagePath + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内

    endTime_pdf2img = datetime.datetime.now()  # 结束时间
    print('pdf转img耗时（s）', (endTime_pdf2img - startTime_pdf2img).seconds)


def recognize_text(image_path):
    # 读取图片
    img = cv2.imread(image_path)

    # 将图片转为灰度图
    # img=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图片的高度和宽度
    height, width = gray.shape

    result = engine.ocr(img)

    text = combine_text(result[0], width)

    return text


def combine_text(orc_result, width):
    left_content = []
    mid_content = []
    right_content = []

    # print(123123123,orc_result)
    # point: [[70.0, 9.0], [109.0, 9.0], [109.0, 23.0], [70.0, 23.0]] 左上 右上 左下 右下

    # 计算最大间隔
    max_right_border = max([x[0][2][0] for x in orc_result])
    min_left_border = min([x[0][2][0] for x in orc_result])
    max_width = max_right_border - min_left_border
    # print(1111111,max_right_border,min_left_border)

    # print(123123123,width,max_width)
    copy_result = orc_result

    try:
        # 获取平均行间距
        height_set = set()
        for elem in orc_result:
            height_set.add(int(elem[0][0][1]))

        avarage_line_gap = 999
        if len(height_set) > 2:
            cur_line_height_list = sorted(list(height_set))[1:-1]
            # print(1111111111,cur_line_height_list)
            sum = 0
            for i in range(1, len(cur_line_height_list)):
                sum += cur_line_height_list[i] - cur_line_height_list[i - 1]
            avarage_line_gap = sum / (len(cur_line_height_list) - 1)

            # avarage_line_gap = sum(cur_line_height_list)/(len(height_set)-2)
            # print(213123123,avarage_line_gap)
            orc_result = sorted(orc_result, key=lambda x: x[0][0][1])
            if abs(orc_result[0][0][0][1] - orc_result[1][0][0][1]) > avarage_line_gap:
                print("删除第0项", orc_result[0])
                orc_result.pop(0)

            if abs(orc_result[-1][0][0][1] - orc_result[-2][0][0][1]) > avarage_line_gap:
                print("删除最后一项", orc_result[-1])
                orc_result.pop(-1)
    except Exception as e:
        print(111111111111, orc_result)
        orc_result = copy_result

    for line in orc_result:
        left_border = line[0][0][0]
        # right_border = line[0][2][0]
        # 计算中心点分栏
        # x_center = (left_border + right_border) / 2
        # or (right_border-left_border)>width*3/4
        if left_border - min_left_border < max_width / 3:
            # 中心点在左边1/3区域，或者右边界减去左边界占了3/4以上，都归到第一栏
            left_content.append(line)
        elif left_border - min_left_border < max_width * 2 / 3:
            mid_content.append(line)
        else:
            right_content.append(line)

    # print(1111,left_content)
    # print(2222,mid_content)
    # print(33333,right_content)

    result = sorted(left_content, key=lambda x: x[0][0][1]) + sorted(mid_content, key=lambda x: x[0][0][1]) + sorted(
        right_content, key=lambda x: x[0][0][1])

    text = [x[1][0] for x in result]
    # for elem in text:
    #     print(11111,elem)

    return '\n'.join(text)


def get_paper_text_info(paper_path, paper_id):
    pdfPath = os.path.join(paper_path, f"{paper_id}.pdf")

    imagePath = os.path.join(paper_path, paper_id)
    # 分割pdf文件
    pyMuPDF_fitz(pdfPath, imagePath)

    file_paths = []
    for root, directories, files in os.walk(imagePath):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    print("pdf页数", len(file_paths))

    result = []
    for _file in file_paths:
        page_result = recognize_text(_file)
        result.append(page_result)
    text = '\n'.join(result)

    if "关键词：" in text:
        index = text.index("关键词：") + 4
    else:
        index = 0

    if "参考文献：" in text:
        index2 = text.index("参考文献：")
    elif "［参考文献］" in text:
        index2 = text.index("［参考文献］")
    else:
        index2 = len(text)

    # todo:全英文摘要 整段去除
    text = text[index:index2]

    return text


def get_file_text_info(file):
    save_path = os.path.join(BASE_DIR, "data/file")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取上传文件的文件名
    filename = file.filename
    # 构造保存文件的路径
    file_path = os.path.join(save_path, filename)

    if str(filename).endswith("jpg") or str(filename).endswith("png"):
        file.save(file_path)
        text = recognize_text(file_path)
    elif str(filename).endswith("pdf"):
        # 保存上传的文件到服务器
        file.save(file_path)
        imagePath = os.path.join(save_path, "img")

        pyMuPDF_fitz(file_path, imagePath)
        file_paths = []
        for root, directories, files in os.walk(imagePath):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        print("pdf页数", len(file_paths))
        result = []
        for _file in file_paths:
            page_result = recognize_text(_file)
            result.append(page_result)
        text = '\n'.join(result)
    else:
        text = ""

    # 删除文件
    shutil.rmtree(save_path)
    return text


def parse_url_ocr_result(url):
    send_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}

    req = requests.get(url, headers=send_headers)  # 通过访问互联网得到文件内容
    bytes_io = io.BytesIO(req.content)  # 转换为字节流

    paper_path = os.path.join(BASE_DIR, f"data/paper")

    if not os.path.exists(paper_path):
        os.makedirs(paper_path)

    save_path = os.path.join(paper_path, "temp.pdf")
    result = bytes_io.getvalue()
    with open(save_path, 'wb') as file:
        file.write(result)  # 保存到本地

    ocr_result = get_paper_text_info(paper_path, "temp")

    # 删除文件
    shutil.rmtree(paper_path)

    return ocr_result


if __name__ == "__main__":
    # 下载文件 1.pdf

    # 读取论文
    paper_id = "1"
    text = get_paper_text_info("./", paper_id)
    print(text)

    # 读取图片
    file_path = os.path.join(BASE_DIR, "1.jpg")
    text = recognize_text(file_path)
    print(text)

    paper_url ="1.pdf"
    result = parse_url_ocr_result(paper_url)
    print(result)

