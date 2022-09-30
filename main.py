from utils import *
import numpy as np


def warpPerspective(img, transform_matrix, output_width, output_height):
    output_matrix = np.zeros((output_width, output_height, 3), dtype='int')
    for i in range(600):
        for j in range(800):
            x = transform_matrix[0][0] * i + transform_matrix[0][1] * j + transform_matrix[0][2]
            y = transform_matrix[1][0] * i + transform_matrix[1][1] * j + transform_matrix[1][2]
            z = transform_matrix[2][0] * i + transform_matrix[2][1] * j + transform_matrix[2][2]
            
            x = int(x // z)
            y = int(y // z)

            if(x < output_width and y < output_height and x >= 0 and y >= 0):
                output_matrix[x][y] = img[i][j] 
    return output_matrix           



def grayScaledFilter(img):
    # Grayscale = (R + G + B / 3)
    gray_transformation = np.array([[0.3, 0.3, 0.3],
                                    [0.3, 0.3, 0.3],
                                    [0.3, 0.3, 0.3]], dtype=np.float)
    output_matrix = Filter(img, gray_transformation)
    return output_matrix

def crazyFilter(img):
    crazy_filter_transformation = np.array([[0, 1, 1],
                                            [1, 0, 0],
                                            [0, 0, 0]], dtype=np.float)
    output_matrix = Filter(img, crazy_filter_transformation)
    return output_matrix

def customFilter(img):
    custom_filter_transformation = np.array([[0.5, 0.5, 0],
                                            [0.5, 0, 0.5],
                                            [0, 0.5, 0.5]])
    output_matrix = Filter(img, custom_filter_transformation)
    # showImage(output_matrix, 'Custom Filter')
    inverted_custom_filter = np.linalg.inv(custom_filter_transformation)
    output_matrix = Filter(output_matrix, inverted_custom_filter)
    # showImage(output_matrix, 'Inverted Custom Filter')


def scaleImg(img, scale_width, scale_height):
    old_width, old_height, _ = img.shape
    output_matrix = np.zeros((scale_width, scale_height, 3), dtype='int')
    x_scale = old_width / scale_width
    y_scale = old_height / scale_height
    for i in range(scale_width):
        for j in range(scale_height):
            output_matrix[i][j] = img[int(i * x_scale)][int(j * y_scale)]
    
    return output_matrix



def cropImg(img, start_row, end_row, start_column, end_column):
    return img[start_column:end_column,start_row:end_row]
    


if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    width, height = 300, 400

    # showImage(image_matrix, title="Input Image")

    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[109, 218], [377, 183], [159, 644], [495, 572]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)
    
    warpedImage = warpPerspective(image_matrix, m, width, height)
    warpedImage = showWarpPerspective(warpedImage)
    # showImage(warpedImage, title='Warp Perspective')

    grayScalePic = grayScaledFilter(warpedImage)
    # showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    # showImage(crazyImage, title="Crazy Filter")

    customFilter(warpedImage)

    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    # showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 600, 400)
    showImage(scaledImage, title="Scaled Image")
