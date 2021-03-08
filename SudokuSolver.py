Import cv2
Import numpy as np
From scipy import ndimage
Import math
Import tensorflow as tf
Import keras
From keras.models import Sequential
From keras.layers import Dense, Dropout, Flatten
From keras.layers import Conv2D, MaxPooling2D
From keras import backend as K
From keras.models import model_from_json
Import sudokuSolver
Import copy


Def write_solution_on_image(image, grid, user_grid):
    SIZE = 9
    Width = image.shape[1] // 9
    Height = image.shape[0] // 9
    For I in range(SIZE):
        For j in range(SIZE):
            If(user_grid[i][j] != 0):    # If user fill this cell
                Continue                # Move on
            Text = str(grid[i][j])
            Off_set_x = width // 15
            Off_set_y = height // 15
            Font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)
        
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width – text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) – math.floor((height – text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), 
                                                  font, font_scale, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
    return image

# Compare every single elements of 2 matrices and return if all corresponding entries are equal
Def two_matrices_are_equal(matrix_1, matrix_2, row, col):
    For I in range(row):
        For j in range(col):
            If matrix_1[i][j] != matrix_2[i][j]:
                Return False
    Return True

# This function is used as the first criteria for detecting whether 
# the contour is a Sudoku board or not: Length of Sides CANNOT be too different (sudoku board is square)
# Return if the longest size is > the shortest size * eps_scale
Def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    Shortest = min(AB, AD, BC, CD)
    Longest = max(AB, AD, BC, CD)
    Return longest > eps_scale * shortest

Def approx_90_degrees(angle, epsilon):
    Return abs(angle – 90) < epsilon

Def largest_connected_component(image):

    Image = image.astype(‘uint8’)
    Nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    Sizes = stats[:, -1]

    If(len(sizes) <= 1):
        Blank_image = np.zeros(image.shape)
        Blank_image.fill(255)
        Return blank_image

    Max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    Max_size = sizes[1]     

    For I in range(2, nb_components):
        If sizes[i] > max_size:
            Max_label = i
            Max_size = sizes[i]

    Img2 = np.zeros(output.shape)
    Img2.fill(255)
    Img2[output == max_label] = 0
    Return img2

# Return the angle between 2 vectors in degrees
Def angle_between(vector_1, vector_2):
    Unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    Unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    Dot_droduct = np.dot(unit_vector_1, unit_vector2)
    Angle = np.arccos(dot_droduct)
    Return angle * 57.2958  # Convert to degree

# Calculate how to centralize the image using its center of mass
Def get_best_shift(img):
    Cy, cx = ndimage.measurements.center_of_mass(img)
    Rows, cols = img.shape
    Shiftx = np.round(cols/2.0-cx).astype(int)
    Shifty = np.round(rows/2.0-cy).astype(int)
    Return shiftx, shifty

# Shift the image using what get_best_shift returns
Def shift(img,sx,sy):
    Rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    Shifted = cv2.warpAffine(img,M,(cols,rows))
    Return shifted
