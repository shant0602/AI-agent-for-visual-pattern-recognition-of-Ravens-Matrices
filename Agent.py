# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops, ImageDraw
import numpy as np
import math
import time


class Agent:

    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    IMAGE_COMPARISON_LIMIT = 98.5
    XOR_IMAGE_COMPARISON_LIMIT = 96
    IMAGE_SIDE = 184
    IMAGE_SIZE = (IMAGE_SIDE, IMAGE_SIDE)
    FUZZY_MATCH_RESOLUTION = IMAGE_SIDE // 30
    FUZZIFICATION_LIMIT = 1  # FUZZY_MATCH_RESOLUTION // 2  # how much to blur the image
    BLACK_WHITE_CUTOFF = 65
    THRESH_PROGRESSION_DIFF = 0.01
    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def __init__(self):
        self.time = time.clock()

    def load_image(self, frame_label, image_path, problem):
        global image_a, image_b, image_c, image_d, image_e, image_f, image_g, \
            image_h, image_1, image_2, image_3, image_4, image_5, image_6, \
            image_7, image_8
        if frame_label == 'A':
            image_a = (Image.open(image_path)).convert('L')
            image_a = self.normalize(image_a)
        if frame_label == 'B':
            image_b = (Image.open(image_path)).convert('L')
            image_b = self.normalize(image_b)
        if frame_label == 'C':
            image_c = (Image.open(image_path)).convert('L')
            image_c = self.normalize(image_c)
        if frame_label == 'D':
            image_d = (Image.open(image_path)).convert('L')
            image_d = self.normalize(image_d)
        if frame_label == '1':
            image_1 = (Image.open(image_path)).convert('L')
            image_1 = self.normalize(image_1)
        if frame_label == '2':
            image_2 = (Image.open(image_path)).convert('L')
            image_2 = self.normalize(image_2)
        if frame_label == '3':
            image_3 = (Image.open(image_path)).convert('L')
            image_3 = self.normalize(image_3)
        if frame_label == '4':
            image_4 = (Image.open(image_path)).convert('L')
            image_4 = self.normalize(image_4)
        if frame_label == '5':
            image_5 = (Image.open(image_path)).convert('L')
            image_5 = self.normalize(image_5)
        if frame_label == '6':
            image_6 = (Image.open(image_path)).convert('L')
            image_6 = self.normalize(image_6)

        if problem.problemType == '3x3':
            if frame_label == 'E':
                image_e = (Image.open(image_path)).convert('L')
                image_e = self.normalize(image_e)

            if frame_label == 'F':
                image_f = (Image.open(image_path)).convert('L')
                image_f = self.normalize(image_f)
            if frame_label == 'G':
                image_g = (Image.open(image_path)).convert('L')
                image_g = self.normalize(image_g)
            if frame_label == 'H':
                image_h = (Image.open(image_path)).convert('L')
                image_h = self.normalize(image_h)
            if frame_label == '7':
                image_7 = (Image.open(image_path)).convert('L')
                image_7 = self.normalize(image_7)
            if frame_label == '8':
                image_8 = (Image.open(image_path)).convert('L')
                image_8 = self.normalize(image_8)

    def normalize(self, image):
        im = image
        im = im.resize(self.IMAGE_SIZE)
        im = self.black_or_white(im)
        return im

    def black_or_white(self, image):
        array = np.asarray(image).copy()  # convert to numpy array
        # Color values are not evenly divided on purpose to emphasize white as a background color
        array[array < self.BLACK_WHITE_CUTOFF] = 0  # Darker colors go to black
        # Lighter colors go to white
        array[array >= self.BLACK_WHITE_CUTOFF] = 255

        return Image.fromarray(array)

    def Solve(self, problem):
        self.time = time.time()
        if problem.problemType == '2x2':
            probDict = problem.figures
            for frame, value in probDict.items():
                figure = probDict[frame]
                frame_image_path = figure.visualFilename
                self.load_image(frame, frame_image_path, problem)
            result = self.affine_transformations(problem)

        elif problem.problemType == '3x3':
            probDict = problem.figures
            for frame, value in probDict.items():
                figure = probDict[frame]
                frame_image_path = figure.visualFilename
                self.load_image(frame, frame_image_path, problem)
            result = self.solve_3x3(problem)
            # result = self.affine_transformations_3(problem)
            # result = self.get_progression_answer(problem)

        # result=self.reflection_across_horizontal(problem)
        # if result==-1:
        #     result=self.affine_transformations(problem)
        # if result==-1:
        #     result=self.reflection_across_vertical(problem)
        # if result==-1:
        #     result=self.rotation_across_horizontal(problem)
        # if result==-1:
        #     result=self.rotation_across_vertical(problem)
        # if result==-1:
        #     result=self.solve_by_pixel_diff(problem)
        self.print_algorithm_time()
        return result

    def solve_3x3(self, problem):
        if self.union() != -1:
            self.print_problem_details(problem, self.union(), 'Union')
            return self.union()
        if self.intersection() != -1:
            self.print_problem_details(
                problem, self.intersection(), 'intersection')
            return self.intersection()
        if self.minus() != -1:
            self.print_problem_details(problem, self.minus(), 'minus')
            return self.minus()
        if self.XOR() != -1:
            self.print_problem_details(problem, self.XOR(), 'XOR')
            return self.XOR()
        # if self.pixel_prog() != -1:
        #     self.print_problem_details(
        #         problem, self.pixel_prog(), 'pixel_progression')
        #     return self.pixel_prog()


        ans = self.isRepeated()
        output = self.allPixelSum()
        # print("ans ",ans)
        # print('output', output)
        prob_answer = [i for i in output if i in ans]
        # print("Probable answers", prob_answer)
        if len(prob_answer) == 1:
            self.print_problem_details(problem, prob_answer[0], ' Region count method')
            return prob_answer[0] 

        if self.dprSum() != -1:
            self.print_problem_details(problem, self.dprSum(), 'Dark pixel ratio sum')
            return self.dprSum()    

        if self.dpr_operation() != -1:
            self.print_problem_details(problem, self.dpr_operation(), 'Dark pixel ratio')
            return self.dpr_operation()

        # if self.get_progression_answer(problem) != -1:
        #     self.print_problem_details(
        #         problem, self.get_progression_answer(problem), 'progression')
        #     return self.get_progression_answer(problem)
        self.print_problem_details(
            problem, -1, 'SKIPPED!')
        self.print_algorithm_time()
        return -1

    def union(self):
        answers = [image_1, image_2, image_3, image_4,
                   image_5, image_6, image_7, image_8]
        # For row
        if self. A_union_B(image_a, image_b, image_c) and \
                self.A_union_B(image_d, image_e, image_f):
            for i, ans in enumerate(answers, 1):
                if self.A_union_B(image_g, image_h, ans):
                    return i
            # For Column
        if self. A_union_B(image_a, image_d, image_g) and \
                self.A_union_B(image_b, image_e, image_h):
            for i, ans in enumerate(answers, 1):
                if self.A_union_B(image_c, image_f, ans):
                    return i
            # Diagonals
        if self. A_union_B(image_f, image_g, image_b) and \
                self.A_union_B(image_h, image_c, image_d):
            for i, ans in enumerate(answers, 1):
                if self.A_union_B(image_a, image_e, ans):
                    return i
        if self. A_union_B(image_f, image_h, image_a) and \
                self.A_union_B(image_g, image_c, image_e):
            for i, ans in enumerate(answers, 1):
                if self.A_union_B(image_b, image_d, ans):
                    return i
        return -1

    def intersection(self):
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]\
            # For row
        if self. A_intersection_B(image_a, image_b, image_c) and \
                self.A_intersection_B(image_d, image_e, image_f):
            for i, ans in enumerate(answers, 1):
                if self.A_intersection_B(image_g, image_h, ans):
                    return i
            # For Column
        if self. A_intersection_B(image_a, image_d, image_g) and \
                self.A_intersection_B(image_b, image_e, image_h):
            for i, ans in enumerate(answers, 1):
                if self.A_intersection_B(image_c, image_f, ans):
                    return i
                # Diagonals
        if self. A_union_B(image_f, image_g, image_b) and \
                self.A_union_B(image_h, image_c, image_d):
            for i, ans in enumerate(answers, 1):
                if self.A_union_B(image_a, image_e, ans):
                    return i
        if self. A_union_B(image_f, image_h, image_a) and \
                self.A_union_B(image_g, image_c, image_e):
            for i, ans in enumerate(answers, 1):
                if self.A_union_B(image_b, image_d, ans):
                    return i

        return -1

    def minus(self):
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]\
            # For row
        if self. A_minus_B(image_a, image_b, image_c) and \
                self.A_minus_B(image_d, image_e, image_f):
            for i, ans in enumerate(answers, 1):
                if self.A_minus_B(image_g, image_h, ans):
                    return i
            # For Column
        if self. A_minus_B(image_a, image_d, image_g) and \
                self.A_minus_B(image_b, image_e, image_h):
            for i, ans in enumerate(answers, 1):
                if self.A_minus_B(image_c, image_f, ans):
                    return i

        return -1

    def XOR(self):
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]\
            # For row
        if self. A_XOR_B(image_a, image_b, image_c) and \
                self.A_XOR_B(image_d, image_e, image_f):
            for i, ans in enumerate(answers, 1):
                if self.A_XOR_B(image_g, image_h, ans):
                    return i
            # For Column
        if self. A_XOR_B(image_a, image_d, image_g) and \
                self.A_XOR_B(image_b, image_e, image_h):
            for i, ans in enumerate(answers, 1):
                if self.A_XOR_B(image_c, image_f, ans):
                    return i
        return -1

    def dpr_operation(self):
        col = []
        row = []
        col_val = []
        row_val = []
        flagCol = False
        flagRow = False
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
        for i, ans in enumerate(answers, 1):
                if self.dpr(image_c, image_f, ans) < 1.0:
                    col.append(i)
                    col_val.append(self.dpr(image_c, image_f, ans))
                    flagCol = True
        if not flagCol:
            resCol = -1
        else:
            resCol = col[col_val.index((min(col_val)))]
            
        for i, ans in enumerate(answers, 1):
                if self.dpr(image_g, image_h, ans) < 1.0:
                    row.append(i)
                    row_val.append(self.dpr(image_g, image_h, ans))
                    flagRow = True
        if not flagRow:
            resRow = -1
        else:
            resRow = row[row_val.index((min(row_val)))]    
        if resRow == resCol:
            return resRow
        elif resRow == -1 or resCol == -1:
            if resRow == -1:
                return resCol
            else:
                return resRow
        else:
            if min(col_val) < min(row_val):
                return resCol
            else:
                return resRow


    def dprSum(self):
        col = []
        row = []
        col_val = []
        row_val = []
        flagCol = False
        flagRow = False
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]

        col1_sum = self.dpr_sum(image_a, image_d, image_g)
        col2_sum = self.dpr_sum(image_b, image_e, image_h)
        avg = (col1_sum + col2_sum)/2
        if abs(col1_sum - col2_sum) <= abs(self.THRESH_PROGRESSION_DIFF*avg):
            for i, ans in enumerate( answers, 1 ):
                col3_sum = self.dpr_sum(image_c, image_f, ans)
                avg2 = (col3_sum + avg)/2
                if abs(col3_sum - avg) <= abs(self.THRESH_PROGRESSION_DIFF*avg2):
                    col.append(i)
                    col_val.append(abs(col3_sum - avg))
                    flagCol = True
        if not flagCol:
            resCol = -1
        else:
            resCol = col[col_val.index((min(col_val)))] 
        

        row1_sum = self.dpr_sum(image_a, image_b, image_c)
        row2_sum = self.dpr_sum(image_d, image_e, image_f)
        avg = (row1_sum + row2_sum)/2
        if abs(row1_sum - row2_sum) <= abs(self.THRESH_PROGRESSION_DIFF*avg):
            for i, ans in enumerate( answers,1 ):
                #ans.show()
                row3_sum = self.dpr_sum(image_g, image_h, ans)
                avg2 = (row3_sum + avg)/2
                if abs(row3_sum - avg) <= abs(self.THRESH_PROGRESSION_DIFF*avg2):
                    row.append(i)
                    row_val.append(abs(row3_sum - avg))
                    flagRow = True
        if not flagRow:
            resRow = -1
        else:
            resRow = row[row_val.index((min(row_val)))]  

        if resRow == resCol:
            return resRow
        elif resRow == -1 or resCol == -1:
            if resRow == -1:
                return resCol
            else:
                return resRow
        else:
            if min(col_val) < min(row_val):
                return resCol
            else:
                return resRow

    def allPixelSum(self):
        col = []
        row = []
        col_val = []
        row_val = []
        flagCol = False
        flagRow = False
        resRow = []
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
        # self.count(im)
        # col1_sum = self.dpr_sum(image_a, image_d, image_g)
        # col2_sum = self.dpr_sum(image_b, image_e, image_h)
        # avg = (col1_sum + col2_sum)/2
        # if abs(col1_sum - col2_sum) <= abs(self.THRESH_PROGRESSION_DIFF*avg):
        #     for i, ans in enumerate( answers, 1 ):
        #         col3_sum = self.dpr_sum(image_c, image_f, ans)
        #         avg2 = (col3_sum + avg)/2
        #         if abs(col3_sum - avg) <= abs(self.THRESH_PROGRESSION_DIFF*avg2):
        #             col.append(i)
        #             col_val.append(abs(col3_sum - avg))
        #             flagCol = True
        # if not flagCol:
        #     resCol = -1
        # else:
        #     resCol = col[col_val.index((min(col_val)))] 
        
        # image_c.show()
        row1Black_sum = self.count_regions(image_a, 'black') + self.count_regions(image_b, 'black') + self.count_regions(image_c, 'black')
        row2Black_sum = self.count_regions(image_d,'black') + self.count_regions(image_e,'black') + self.count_regions(image_f, 'black')
        row1White_sum = self.count_regions(image_a) + self.count_regions(image_b) + self.count_regions(image_c)
        row2White_sum = self.count_regions(image_d) + self.count_regions(image_e) + self.count_regions(image_f)

        if row1Black_sum == row2Black_sum and row1White_sum == row2White_sum:
            ghBlack_Sum = self.count_regions(image_g, 'black') + self.count_regions(image_h, 'black')
            ghWhite_Sum = self.count_regions(image_g) + self.count_regions(image_h)
            for i, ans in enumerate( answers,1 ):
                if row1Black_sum == ghBlack_Sum + self.count_regions(ans,'black') and \
                row1White_sum == ghWhite_Sum + self.count_regions(ans):
                    resRow.append(i)
                    flagRow = True           
        
        if len(resRow) > 0:
            return resRow
        else:
            return [0]

    def isRepeated(self):
        probable_answer = []
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
        questions = [image_a, image_b, image_c, image_d, image_e, image_f, image_g, image_h]
        flagMatch = False
        for i, q in enumerate(questions):
            if self.match(q, questions[i+1:]):
                flagMatch = True
        if not flagMatch:
            for ans in answers:
                if not self.match(ans, questions):
                    probable_answer.append(answers.index(ans) + 1)
        if len(probable_answer) > 0:
            return probable_answer
        else:
            return [-1]

    def match(self, im, imCollection):
        for i in imCollection:
            if self.sim(im, i, True) > self.IMAGE_COMPARISON_LIMIT:
                return True
        return False

    def count_regions(self, im1, color = 'white' ):
        im_array = np.asarray(im1)
        im_array.flags.writeable = True
        size = np.shape(im_array)
        if color == 'black':
            regionType = 0
        elif color == 'white':
            regionType = 255
        regionCount = 0
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                if im_array[i][j] == regionType:
                    regionCount += 1
                    self.findNeighbors(im_array, i, j, regionType)
        return regionCount

    def findNeighbors(self, im_array, row, col, regionType):
        junk_value = 1000
        stack = [(row, col)]
        im_array[row][col] = junk_value

        while len(stack) != 0 :
            loc = stack.pop()
            row = loc[0]
            col = loc[1]

            if (col-1) > -1 and im_array[row][col-1] == regionType:
                stack.append((row ,col-1))
                im_array[row][col-1] = junk_value

            if (col+1) < len(im_array) and im_array[row][col+1] == regionType:
                stack.append((row ,col+1))
                im_array[row][col+1] = junk_value

            if (row-1) > -1 and im_array[row -1][col] == regionType:
                stack.append((row - 1,col))
                im_array[row - 1][col] = junk_value

            if (row+1) < len(im_array) and im_array[row + 1][col] == regionType:
                stack.append((row + 1,col))
                im_array[row + 1][col] = junk_value

    def pixel_prog(self):
        answers = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]\
            # For row
        if self.pixel_progression(image_a, image_b, image_c) and \
                self.pixel_progression(image_d, image_e, image_f):
            for i, ans in enumerate(answers, 1):
                if self.pixel_progression(image_g, image_h, ans):
                    return i
            # For Column
        if self.pixel_progression(image_a, image_d, image_g) and \
                self.pixel_progression(image_b, image_e, image_h):
            for i, ans in enumerate(answers, 1):
                if self.pixel_progression(image_c, image_f, ans):
                    return i

        return -1

    def reflection_across_horizontal(self, problem):
        image_a = Image.open(problem.figures['A'].visualFilename)
        image_b = Image.open(problem.figures['B'].visualFilename)
        image_c = Image.open(problem.figures['C'].visualFilename)
        trans_a = image_a.transpose(Image.FLIP_LEFT_RIGHT)
        image_diff = self.find_percent_difference(trans_a, image_b)

        if image_diff < 2:
            trans_c = image_c.transpose(Image.FLIP_LEFT_RIGHT)
            percent_diff_array = []
            for i in range(1, 7):
                image_options = Image.open(
                    problem.figures[str(i)].visualFilename)
                options_diff = self.find_percent_difference(
                    trans_c, image_options)
                percent_diff_array.append(options_diff)
            if min(percent_diff_array) < 5:
                print(problem.name, " Answer is", percent_diff_array.index(
                    min(percent_diff_array)) + 1)
                return percent_diff_array.index(min(percent_diff_array)) + 1

        return -1

    def reflection_across_vertical(self, problem):
        image_a = Image.open(problem.figures['A'].visualFilename)
        image_b = Image.open(problem.figures['B'].visualFilename)
        image_c = Image.open(problem.figures['C'].visualFilename)
        trans_a = image_a.transpose(Image.FLIP_TOP_BOTTOM)
        image_diff = self.find_percent_difference(trans_a, image_c)

        if image_diff < 1:
            trans_b = image_b.transpose(Image.FLIP_TOP_BOTTOM)
            percent_diff_array = []
            for i in range(1, 7):
                image_options = Image.open(
                    problem.figures[str(i)].visualFilename)
                options_diff = self.find_percent_difference(
                    trans_b, image_options)
                percent_diff_array.append(options_diff)
            if min(percent_diff_array) < 5:
                print(problem.name, " Answer is", percent_diff_array.index(
                    min(percent_diff_array)) + 1)
                return percent_diff_array.index(min(percent_diff_array)) + 1

        return -1

    def rotation_across_horizontal(self, problem):
        image_a = Image.open(problem.figures['A'].visualFilename)
        image_b = Image.open(problem.figures['B'].visualFilename)
        image_c = Image.open(problem.figures['C'].visualFilename)

        rot_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for i in range(0, len(rot_angles), 1):
            rotate_a = image_a.rotate(rot_angles[i])
            image_diff = self.find_percent_difference(rotate_a, image_b)
            if image_diff < 1:
                rotate_c = image_c.rotate(rot_angles[i])
                percent_diff_array = []
                for j in range(1, 7):
                    image_options = Image.open(
                        problem.figures[str(j)].visualFilename)
                    option_diff = self.find_percent_difference(
                        rotate_c, image_options)
                    percent_diff_array.append(option_diff)
                if min(percent_diff_array) < 3:
                    print(problem.name, " Answer is", percent_diff_array.index(
                        min(percent_diff_array)) + 1)
                    return percent_diff_array.index(min(percent_diff_array)) + 1
        return -1

    def rotation_across_vertical(self, problem):
        image_a = Image.open(problem.figures['A'].visualFilename)
        image_b = Image.open(problem.figures['B'].visualFilename)
        image_c = Image.open(problem.figures['C'].visualFilename)

        rot_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for i in range(0, len(rot_angles), 1):
            rotate_a = image_a.rotate(rot_angles[i])
            image_diff = self.find_percent_difference(rotate_a, image_c)
            if image_diff < 1:
                rotate_b = image_b.rotate(rot_angles[i])
                percent_diff_array = []
                for j in range(1, 7):
                    image_options = Image.open(
                        problem.figures[str(j)].visualFilename)
                    option_diff = self.find_percent_difference(
                        rotate_b, image_options)
                    percent_diff_array.append(option_diff)
                if min(percent_diff_array) < 3:
                    print(problem.name, " Answer is", percent_diff_array.index(
                        min(percent_diff_array)) + 1)
                    return percent_diff_array.index(min(percent_diff_array)) + 1
        return -1

    def print_algorithm_time(self):
        algorithm_time = time.time() - self.time
        print()
        print('Solution time = ', int(algorithm_time * 1000), 'milliseconds')
        self.time = time.time()

 
    def affine_transformations(self, problem):
        sim_matrix = {}
        similarity = []
        operation = []
        # Form the similarity matrix
        self.base_unary_trans(image_a, image_b, sim_matrix,
                              "Row-wise", "None", "2x2")
        self.base_unary_trans(image_a, image_c, sim_matrix,
                              "Column-wise", "None", "2x2")

        for key, value in sim_matrix.items():
            similarity.append(value[0])
            operation.append(key)

        # for key,value in sim_matrix.items():
        #     print(key, ":                        ",value)
        index_max_s = similarity.index(max(similarity))

        # Get a transformation for the maximum similarity transformation
        AminusB = sim_matrix[operation[index_max_s]][1]
        BminusA = sim_matrix[operation[index_max_s]][2]
        relation = sim_matrix[operation[index_max_s]][3]
        angle = sim_matrix[operation[index_max_s]][4]
        flip = sim_matrix[operation[index_max_s]][5]

        # print(sim_matrix[operation[index_max_s]][0])
        # print(relation," ",angle," ",flip)
        # T(A)
        result = self.threshold_image_white(image_a.rotate(angle))
        if flip == "Vertical flip":
            result = self.threshold_image_white(
                result.transpose(Image.FLIP_TOP_BOTTOM))
        elif flip == "Horizontal flip":
            result = self.threshold_image_white(
                result.transpose(Image.FLIP_LEFT_RIGHT))
        # result.show()
        # Apply image composition operand
        if AminusB <= BminusA:
            if relation == "Row-wise":
                X = self.threshold_image_white(
                    self.difference_image(image_b, result))
            else:
                X = self.threshold_image_white(
                    self.difference_image(image_c, result))
        elif AminusB > BminusA:
            if relation == "Row-wise":
                X = self.threshold_image_white(
                    self.difference_image(result, image_b))
            else:
                X = self.threshold_image_white(
                    self.difference_image(result, image_c))
        # result.save('A of output.png')
        # image_c.save('C of output.png')
        # X.show()
        # T(Z=C or B)

        if relation == "Row-wise":
            result = self.threshold_image_white(image_c.rotate(angle))
            if flip == "Vertical flip":
                result = self.threshold_image_white(
                    result.transpose(Image.FLIP_TOP_BOTTOM))
            elif flip == "Horizontal flip":
                result = self.threshold_image_white(
                    result.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            result = self.threshold_image_white(image_b.rotate(angle))
            if flip == "Vertical flip":
                result = self.threshold_image_white(
                    result.transpose(Image.FLIP_TOP_BOTTOM))
            elif flip == "Horizontal flip":
                result = self.threshold_image_white(
                    result.transpose(Image.FLIP_LEFT_RIGHT))
        # result.show()

        if AminusB < BminusA:
            res = self.addition_image(result, X)
        else:
            res = self.difference_image(result, X)
        # res.show()
        s = []
        option_image = [image_1, image_2, image_3, image_4, image_5, image_6]
        for im in option_image:
            s.append(self.sim(res, im))
        print(max(s))
        print(problem.name, " Answer is", s.index(max(s)) + 1)

        # option_image[s.index(max(s))].show()
        return s.index(max(s)) + 1

    def affine_transformations_3(self, problem):
        sim_matrix = {}
        similarity = []
        operation = []
        ImageColl = {"A": image_a, "B": image_b, "C": image_c, "D": image_d,
                     "E": image_e, "F": image_f, "G": image_g, "H": image_h}
        # Form the similarity matrix 2x2 combinations
        # self.base_unary_trans(image_b, image_c, sim_matrix,"Row-wise", "H", "2x2")
        # self.base_unary_trans(image_e, image_f, sim_matrix,"Row-wise", "H", "2x2")
        # self.base_unary_trans(image_g, image_h, sim_matrix,"Row-wise", "H", "2x2")
        # self.base_unary_trans(image_a, image_c, sim_matrix,"Row-wise", "G", "2x2")
        # self.base_unary_trans(image_d, image_f, sim_matrix,"Row-wise", "G", "2x2")
        # # stage 2 Rowise 2x2
        # # self.base_unary_trans(image_a, image_b, sim_matrix,"Row-wise", "H", "2x2")
        # # self.base_unary_trans(image_d, image_e, sim_matrix,"Row-wise", "H", "2x2")

        # self.base_unary_trans(image_d, image_g, sim_matrix,"Column-wise", "F", "2x2")
        # self.base_unary_trans(image_e, image_h, sim_matrix,"Column-wise", "F", "2x2")
        # self.base_unary_trans(image_c, image_f, sim_matrix,"Column-wise", "F", "2x2")
        # self.base_unary_trans(image_a, image_g, sim_matrix,"Column-wise", "C", "2x2")
        # self.base_unary_trans(image_b, image_h, sim_matrix,"Column-wise", "C", "2x2")
    #     #stage 2 Columnwise 2x2
    #     self.base_unary_trans(image_a, image_d, sim_matrix,"Column-wise", "F", "2x2")
    #     self.base_unary_trans(image_b, image_e, sim_matrix,"Column-wise", "F", "2x2")
    #    #stage 4 Diagonal-1 2x2
    #     self.base_unary_trans(image_f, image_g, sim_matrix,"Diaganol-wise-1", "E", "2x2")
    #     self.base_unary_trans(image_g, image_b, sim_matrix,"Diaganol-wise-1", "E", "2x2")
    #     self.base_unary_trans(image_h, image_c, sim_matrix,"Diaganol-wise-1", "E", "2x2")
    #     self.base_unary_trans(image_c, image_d, sim_matrix,"Diaganol-wise-1", "E", "2x2")
    #     self.base_unary_trans(image_a, image_e, sim_matrix,"Diaganol-wise-1", "E", "2x2")
    #     self.base_unary_trans(image_f, image_b, sim_matrix,"Diaganol-wise-1", "A", "2x2")
    #     self.base_unary_trans(image_h, image_d, sim_matrix,"Diaganol-wise-1", "A", "2x2")
    #    #stage 4 Diagonal-2 2x2
    #     self.base_unary_trans(image_f, image_h, sim_matrix,"Diaganol-wise-2", "D", "2x2")
    #     self.base_unary_trans(image_h, image_a, sim_matrix,"Diaganol-wise-2", "D", "2x2")
    #     self.base_unary_trans(image_g, image_c, sim_matrix,"Diaganol-wise-2", "D", "2x2")
    #     self.base_unary_trans(image_c, image_e, sim_matrix,"Diaganol-wise-2", "D", "2x2")
    #     self.base_unary_trans(image_b, image_d, sim_matrix,"Diaganol-wise-2", "D", "2x2")
    #     self.base_unary_trans(image_f, image_a, sim_matrix,"Diaganol-wise-2", "B", "2x2")
    #     self.base_unary_trans(image_g, image_e, sim_matrix,"Diaganol-wise-2", "B", "2x2")
    #     self.base_unary_trans(image_f, image_h, sim_matrix,"Diaganol-wise-2", "D", "2x2")
    #     self.base_unary_trans(image_f, image_h, sim_matrix,"Diaganol-wise-2", "D", "2x2")

        # Form the similarity matrix 3x3 combinations
        self.base_binary_trans(image_a, image_b, image_c,
                               sim_matrix, "Row-wise", "A", "B",)
        self.base_binary_trans(image_d, image_e, image_f,
                               sim_matrix, "Row-wise", "D", "E")
        self.base_binary_trans(image_a, image_d, image_g,
                               sim_matrix, "Column-wise", "A", "D")
        self.base_binary_trans(image_b, image_e, image_h,
                               sim_matrix, "Column-wise", "B", "E")
        # #3x3 diaganols stage 3
        # self.base_binary_trans(image_f, image_g, image_b, sim_matrix,"Diaganol-wise-1", "F", "G",)
        # self.base_binary_trans(image_h, image_c, image_d, sim_matrix,"Diaganol-wise-1", "H", "C")
        # self.base_binary_trans(image_f, image_h, image_a, sim_matrix,"Diaganol-wise-2", "F", "H")
        # self.base_binary_trans(image_g, image_c, image_e, sim_matrix,"Diaganol-wise-2", "G", "C")

        for key, value in sim_matrix.items():
            similarity.append(value[0])
            operation.append(key)
        # Printing the sim matrix
        # for key,value in sim_matrix.items():
        #     print(key, ":                        ",value)
        index_max_s = similarity.index(max(similarity))
        print(operation[index_max_s])
        print(sim_matrix[operation[index_max_s]])
        # Get a transformation for the maximum similarity transformation
        AminusB = sim_matrix[operation[index_max_s]][1]
        BminusA = sim_matrix[operation[index_max_s]][2]
        relation = sim_matrix[operation[index_max_s]][3]
        angle = sim_matrix[operation[index_max_s]][4]
        flip = sim_matrix[operation[index_max_s]][5]
        print("Maximum S operation is: ",
              sim_matrix[operation[index_max_s]][0])
        # print(relation," ",angle," ",flip)
        # T(A)

        result = sim_matrix[operation[index_max_s]][6]
        last_image = sim_matrix[operation[index_max_s]][7]
        correspond = sim_matrix[operation[index_max_s]][8]
        for key, value in ImageColl.items():
            if key == correspond:
                correspond = value
                break
        typeMat = sim_matrix[operation[index_max_s]][9]
        operator1 = operation[index_max_s]
        operator = operator1.split("-")[1]
        # if operator != 'progression':
        #     result.show()
        if operator != 'progression':
            # Apply image composition operand
            if AminusB <= BminusA:
                X = self.threshold_image_white(
                    self.difference_image(last_image, result))
            elif AminusB > BminusA:
                X = self.threshold_image_white(
                    self.difference_image(result, last_image))
            # result.save('A of output.png')
            # image_c.save('C of output.png')
            # X.show()
        # T(Z=C or B)
        if typeMat == '2x2':
            result = self.threshold_image_white(correspond.rotate(angle))
            if flip == "Vertical flip":
                result = self.threshold_image_white(
                    result.transpose(Image.FLIP_TOP_BOTTOM))
            elif flip == "Horizontal flip":
                result = self.threshold_image_white(
                    result.transpose(Image.FLIP_LEFT_RIGHT))
        elif typeMat == '3x3':
            if relation == "Row-wise":
                if operator == "union":
                    result = self.threshold_image_white(
                        ImageChops.darker(image_g, image_h))
                elif operator == "intersection":
                    result = self.threshold_image_white(
                        ImageChops.lighter(image_g, image_h))
                elif operator == "minus1":
                    result = self.difference_image(image_g, image_h)
                elif operator == "minus1":
                    result = self.difference_image(image_h, image_g)
                elif operator == "XOR":
                    result = self.XOR_image(image_g, image_h)
                elif operator == 'progression':
                    diff1 = self.percent_diff(image_g, image_h)
                    option_image = [image_1, image_2, image_3,
                                    image_4, image_5, image_6, image_7, image_8]
                    answer = 1
                    for im in option_image:
                        diff2 = self.percent_diff(image_h, im)
                        avg_diff = (diff1 + diff2)/2.0
                        if abs(diff1 - avg_diff) <= abs(self.THRESH_PROGRESSION_DIFF*avg_diff):
                            self.print_problem_details(
                                problem, answer, operator)
                            return answer
                        answer = answer + 1
                        if answer == 9:
                            print("progression skipped")
                            return -1
            elif relation == "Column-wise":
                if operator == "union":
                    result = self.threshold_image_white(
                        ImageChops.darker(image_c, image_f))
                elif operator == "intersection":
                    result = self.threshold_image_white(
                        ImageChops.lighter(image_c, image_f))
                elif operator == "minus1":
                    result = self.difference_image(image_c, image_f)
                elif operator == "minus1":
                    result = self.difference_image(image_f, image_c)
                elif operator == "XOR":
                    result = self.XOR_image(image_c, image_f)
                elif operator == 'progression':
                    diff1 = self.percent_diff(image_c, image_f)
                    option_image = [image_1, image_2, image_3,
                                    image_4, image_5, image_6, image_7, image_8]
                    answer = 1
                    for im in option_image:
                        diff2 = self.percent_diff(image_f, im)
                        avg_diff = (diff1 + diff2)/2.0
                        if abs(diff1 - avg_diff) <= abs(self.THRESH_PROGRESSION_DIFF*avg_diff):
                            self.print_problem_details(
                                problem, answer, operator)
                            return answer
                        answer = answer + 1
                        if answer == 9:
                            print("progression skipped")
                            return -1
                    # stage 3
            elif relation == "Diaganol-wise-1":
                if operator == "union":
                    result = self.threshold_image_white(
                        ImageChops.darker(image_a, image_e))
                elif operator == "intersection":
                    result = self.threshold_image_white(
                        ImageChops.lighter(image_a, image_e))
                elif operator == "minus1":
                    result = self.difference_image(image_a, image_e)
                elif operator == "minus1":
                    result = self.difference_image(image_e, image_a)
                elif operator == "XOR":
                    result = self.XOR_image(image_a, image_e)
            elif relation == "Diaganol-wise-2":
                if operator == "union":
                    result = self.threshold_image_white(
                        ImageChops.darker(image_b, image_d))
                elif operator == "intersection":
                    result = self.threshold_image_white(
                        ImageChops.lighter(image_b, image_d))
                elif operator == "minus1":
                    result = self.difference_image(image_b, image_d)
                elif operator == "minus1":
                    result = self.difference_image(image_d, image_b)
                elif operator == "XOR":
                    result = self.XOR_image(image_b, image_d)
        # result.show()

        if AminusB < BminusA:
            res = self.addition_image(result, X)
        else:
            res = self.difference_image(result, X)
        # res.show()
        s = []
        option_image = [image_1, image_2, image_3,
                        image_4, image_5, image_6, image_7, image_8]
        for im in option_image:
            s.append(self.sim(res, im))
        print(max(s))
        self.print_problem_details(problem, s.index(max(s)) + 1, operator)
        # option_image[s.index(max(s))].show()
        return s.index(max(s)) + 1

    def base_unary_trans(self, first_image, second_image, matrix_similarity, relation, correspond, typeMat):
        image_1 = self.threshold_image_white(first_image.convert('L'))
        image_2 = self.threshold_image_white(second_image.convert('L'))

        rot_angles = [0, 90, 180, 270]
        for angle in rot_angles:
            image_1_rot = image_1.rotate(angle)
            image_1_rot = self.threshold_image_white(image_1_rot)
            s = self.sim(image_1_rot, image_2, True)
            name = "rotate"+str(angle)+" "+relation
            matrix_similarity[name] = [s, self.f_difference_image(image_1_rot, image_2),
                                       self.f_difference_image(
                                           image_2, image_1_rot), relation, angle, "No", image_1_rot,
                                       image_2, correspond, typeMat]

            if angle != 180 and angle != 270:
                image_1_rot_vertical_flip = image_1_rot.transpose(
                    Image.FLIP_TOP_BOTTOM)
                image_1_rot_vertical_flip = self.threshold_image_white(
                    image_1_rot_vertical_flip)
                s = self.sim(image_1_rot_vertical_flip, image_2, True)
                name = "rotate"+str(angle)+"-vertical flip"+" "+relation
                matrix_similarity[name] = [s, self.f_difference_image(image_1_rot_vertical_flip, image_2),
                                           self.f_difference_image(
                                               image_2, image_1_rot_vertical_flip),
                                           relation, angle, "Vertical flip", image_1_rot_vertical_flip, image_2, correspond,
                                           typeMat]

                image_1_rot_horizontal_flip = image_1_rot.transpose(
                    Image.FLIP_LEFT_RIGHT)
                image_1_rot_horizontal_flip = self.threshold_image_white(
                    image_1_rot_horizontal_flip)
                s = self.sim(image_1_rot_horizontal_flip, image_2, True)
                name = "rotate"+str(angle)+"-horizontal flip"+" "+relation
                matrix_similarity[name] = [s, self.f_difference_image(image_1_rot_horizontal_flip, image_2),
                                           self.f_difference_image(
                                               image_2, image_1_rot_horizontal_flip),
                                           relation, angle, "Horizontal flip", image_1_rot_horizontal_flip, image_2, correspond,
                                           typeMat]

    def base_binary_trans(self, first_image, second_image, third_image, matrix_similarity, relation, im1, im2):
        image_11 = self.threshold_image_white(first_image)
        image_22 = self.threshold_image_white(second_image)
        image_33 = self.threshold_image_white(third_image)

        image_AunB = ImageChops.darker(image_11, image_22)
        image_AunB = self.threshold_image_white(image_AunB)
        s = self.sim(image_AunB, image_33, True)
        # if im1 == 'D' and im2 == 'E':
        #     image_11.show()
        #     image_22.show()
        #     image_33.show()
        #     image_11.save("DD.png","")
        #     image_22.save("EE.png", "")
        #     image_33.save("FF.png", "")
        #     image_AunB.show()
        #     print("AunB")
        #     print(s)
        name = im1+"-union-"+im2+"-"+relation
        matrix_similarity[name] = [s, self.f_difference_image(image_AunB, image_33),
                                   self.f_difference_image(
                                       image_33, image_AunB), relation, 0, "No", image_AunB, image_33,
                                   "None", "3x3"]

        image_AintB = ImageChops.darker(image_11, image_22)
        image_AintB = self.threshold_image_white(image_AintB)
        s = self.sim(image_AintB, image_33, True)
        name = im1+"-intersection-"+im2+"-"+relation
        matrix_similarity[name] = [s, self.f_difference_image(image_AintB, image_33),
                                   self.f_difference_image(
                                       image_33, image_AintB), relation, 0, "No", image_AintB, image_33,
                                   "None", "3x3"]

        image_AminusB = self.difference_image(image_11, image_22)
        s = self.sim(image_AminusB, image_33)
        name = im1+"-minus1-"+im2+"-"+relation
        matrix_similarity[name] = [s, self.f_difference_image(image_AminusB, image_33),
                                   self.f_difference_image(
                                       image_33, image_AminusB), relation, 0, "No", image_AminusB, image_33,
                                   "None", "3x3"]

        image_BminusA = self.difference_image(image_22, image_11)
        s = self.sim(image_BminusA, image_33, True)
        name = im2+"-minus2-"+im1+"-"+relation
        matrix_similarity[name] = [s, self.f_difference_image(image_BminusA, image_33),
                                   self.f_difference_image(
                                       image_33, image_BminusA), relation, 0, "No", image_BminusA, image_33,
                                   "None", "3x3"]

        image_A_XOR_B = self.XOR_image(image_11, image_22)
        s = self.sim(image_A_XOR_B, image_33)
        name = im1+"-XOR-"+im2+"-"+relation
        matrix_similarity[name] = [s, self.f_difference_image(image_A_XOR_B, image_33),
                                   self.f_difference_image(
                                       image_33, image_A_XOR_B), relation, 0, "No", image_A_XOR_B, image_33,
                                   "None", "3x3"]

        s = self.pixel_progression(image_11, image_22, image_33)
        name = im1+"-progression-"+im2+"-"+relation
        matrix_similarity[name] = [s, 0, 0, relation,
                                   0, "No", "None", "None", "None", "3x3"]

    def sim(self, im1, im2, displace_check=False):
        if displace_check:
            im1, im2 = self.displace_check_comp(im1, im2)
        res = ImageChops.invert(ImageChops.difference(im1, im2))
        total_pixels = res.size[0] * res.size[1]
        return (np.count_nonzero(res) / total_pixels) * 100

    def displace_check_comp(self, im1, im2):
        up, down, left, right = True, True, True, True

        improvements = 0
        max_score = self.sim(im1, im2)
        while True:
            offset_image = None
            search_direction = None

            # offset the image
            if up:
                offset_image = ImageChops.offset(im2, 0, -1)
                search_direction = 'UP'
            elif down:
                offset_image = ImageChops.offset(im2, 0, 1)
                search_direction = 'DOWN'
            elif left:
                offset_image = ImageChops.offset(im2, -1, 0)
                search_direction = 'LEFT'
            elif right:
                offset_image = ImageChops.offset(im2, 1, 0)
                search_direction = 'RIGHT'
            else:
                break

            # test the offset image to see if it's better
            score = self.sim(im1, offset_image)
            if score > max_score and improvements <= self.FUZZY_MATCH_RESOLUTION:  # if so, update im2
                im2 = offset_image
                max_score = score
                improvements += 1
            # print('Improved', search_direction)
            else:  # if not, turn off whatever step we just took
                # print('Failed to improve', search_direction)
                if search_direction == 'UP':
                    up = False
                elif search_direction == 'DOWN':
                    down = False
                elif search_direction == 'LEFT':
                    left = False
                elif search_direction == 'RIGHT':
                    right = False

                improvements = 0

        return [im1, im2]

    def f_union_image(self, first_image, second_image):
        width1, height1 = first_image.size
        width2, height2 = second_image.size
        if width1 == width2 and height1 == height2:
            union = self.threshold_image_white(
                ImageChops.darker(first_image, second_image))
            union = ImageChops.invert(union)
            union_array = self.normalize_array(union)
            sum = 0
            for i in range(height1):
                for j in range(width1):
                    sum += union_array[i, j]
            return sum
        else:
            print("Agent:f_union_image::The two images differ in size")

    def f_intersection_image(self, first_image, second_image):
        width1, height1 = first_image.size
        width2, height2 = second_image.size
        if width1 == width2 and height1 == height2:
            intersection = self.threshold_image_white(
                ImageChops.lighter(first_image, second_image))
            intersection = ImageChops.invert(intersection)
            intersection_array = self.normalize_array(intersection)
            sum = 0
            for i in range(height1):
                for j in range(width1):
                    sum += intersection_array[i, j]
            return sum
        else:
            print("Agent:f_intersection_image::The two images differ in size")

    def f_difference_image(self, first_image, second_image):
        width1, height1 = first_image.size
        width2, height2 = second_image.size

        if width1 == width2 and height1 == height2:
            difference = ImageChops.difference(
                first_image, ImageChops.lighter(first_image, second_image))
            difference = self.threshold_image(difference)
            diff_array = self.normalize_array(difference)
            sum = 0
            for i in range(height1):
                for j in range(width1):
                    sum += diff_array[i, j]
            return sum
        else:
            print("Agent:f_difference_image::The two images differ in size")

    def A_union_B(self, first_image, second_image, third_image):
        union = ImageChops.darker(first_image, second_image)
        union = self.threshold_image_white(union)
        simMeasure = self.sim(union, third_image, True)
        if simMeasure > self.IMAGE_COMPARISON_LIMIT:
            return True
        else:
            return False

    def A_intersection_B(self, first_image, second_image, third_image):
        intersection = ImageChops.lighter(first_image, second_image)
        intersection = self.threshold_image_white(intersection)
        if self.sim(intersection, third_image, True) > self.IMAGE_COMPARISON_LIMIT:
            return True
        else:
            return False

    def A_minus_B(self, first_image, second_image, third_image):
        minus = ImageChops.invert(
            ImageChops.difference(first_image, second_image))
        minus = self.threshold_image_white(minus)
        if self.sim(minus, third_image, True) > self.IMAGE_COMPARISON_LIMIT:
            return True
        else:
            return False

    def A_XOR_B(self, first_image, second_image, third_image):
        AunB = ImageChops.darker(first_image, second_image)
        AintB = ImageChops.lighter(first_image, second_image)
        difference = self.difference_image(AunB, AintB)
        XOR = self.threshold_image_white(difference)
        XOR_Value = self.sim(XOR, third_image, True)
        if XOR_Value > self.XOR_IMAGE_COMPARISON_LIMIT:
            return True
        else:
            return False

    def XOR_image(self, im1, im2):
        first_image = im1
        second_image = im2
        width1, height1 = first_image.size
        width2, height2 = second_image.size

        if width1 == width2 and height1 == height2:
            AunB = ImageChops.darker(first_image, second_image)
            AintB = ImageChops.lighter(first_image, second_image)
            difference = self.difference_image(AunB, AintB)
            difference = self.threshold_image_white(difference)
            return difference
        else:
            print("Agent:difference_image::The two images differ in size")

    def difference_image(self, im1, im2):

        first_image = im1
        second_image = im2
        width1, height1 = first_image.size
        width2, height2 = second_image.size

        if width1 == width2 and height1 == height2:
            difference = ImageChops.difference(
                first_image, ImageChops.lighter(first_image, second_image))
            difference = self.threshold_image(difference)
            return self.threshold_image_white(ImageChops.invert(difference))
        else:
            print("Agent:difference_image::The two images differ in size")

    def addition_image(self, im1, im2):
        first_image = im1
        second_image = im2
        width1, height1 = first_image.size
        width2, height2 = second_image.size

        if width1 == width2 and height1 == height2:
            im = ImageChops.darker(first_image, second_image)
            self.threshold_image_white(im)
            return im
        else:
            print("Agent:addition_image::The two images differ in size")

    def normalize_array(self, image):
        width1, height1 = image.size
        newAr = np.array(image, dtype='int64')
        for i in range(height1):
            for j in range(width1):
                newAr[i, j] = newAr[i, j]/255.0
        return newAr

    def threshold_image(self, image):
        width1, height1 = image.size
        newAr = np.array(image, dtype='int64')
        for i in range(height1):
            for j in range(width1):
                if(newAr[i, j] < 127.5):
                    newAr[i, j] = 0
        return Image.fromarray(np.uint8(newAr), 'L')

    def threshold_image_white(self, image):
        width1, height1 = image.size
        newAr = np.array(image, dtype='int64')
        for i in range(height1):
            for j in range(width1):
                if(newAr[i, j] > 127.5):
                    newAr[i, j] = 255
        return Image.fromarray(np.uint8(newAr), 'L')

    def find_percent_difference(self, first_image, second_image):

      # Reference: http://rosettacode.org/wiki/Percentage_difference_between_images#Python

        pairs = zip(first_image.getdata(), second_image.getdata())
        if len(first_image.getbands()) == 1:
            # for gray-scale jpegs
            dif = sum(abs(p1 - p2) for p1, p2 in pairs)
        else:
            dif = sum(abs(c1 - c2)
                      for p1, p2 in pairs for c1, c2 in zip(p1, p2))

        n_components = first_image.size[0] * first_image.size[1] * 3

        return (dif / 255.0 * 100) / n_components

    # Returns the percentage of non-zero pixels in the image
    def percent_diff(self, im1, im2):
        total_pixels = im1.size[0] * im1.size[1]
        return (self.count(im1, 'black') - self.count(im2, 'black'))*100 / total_pixels

    # Returns a count of the non-zero (white) pixels in the image
    def count(self, im, color='white'):
        if color == 'black':
            im = ImageChops.invert(im)
        return np.count_nonzero(im)

    def black_match_rate(self, im1, im2):
        im1_black = self.count(im1, 'black')
        im2_black = self.count(im2, 'black')
        total = im1_black + im2_black
        changed = self.count(ImageChops.difference(im1, im2))
        same = (total-changed)/2
        return same/max(im1_black, im2_black)*100
        # Answer method that tests problem for A-B=C pixel summation pattern

    def pixel_progression(self, im1, im2, im3):
        sim_value = 0.0
        im_a = im1
        im_b = im2
        im_c = im3
        diff1 = self.percent_diff(im_a, im_b)
        diff2 = self.percent_diff(im_b, im_c)
        avg_diff = (diff1 + diff2)/2.0

        if abs(diff1 - avg_diff) <= abs(self.THRESH_PROGRESSION_DIFF*avg_diff):
            return True
        else:
            return False
            
    def dpr(self, im1, im2, im3):
        diff1 = self.percent_diff(im1, im2)
        diff2 = self.percent_diff(im2, im3)
        return abs(diff1 - diff2)

    def dpr_sum(self, im1, im2, im3):
        total_pixels = im1.size[0] * im1.size[1]
        return (self.count(im1, 'black') + self.count(im2, 'black') + self.count(im3, 'black'))*100 / total_pixels


    def print_problem_details(self, problem, answer, operator):
        print()
        print('==================================================')
        print(problem.name, '(' + problem.problemType + ')', ' Solved by ' +
              operator, ' answer is : ', answer)
        print('==================================================')
