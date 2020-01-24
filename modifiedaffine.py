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
# import numpy

import TransformationFile as Pillow
import Transform as Trans
import os
import sys
import time
import math


class Agent:

	submitting = True
	DEVIATION_DIFFERENCE_REQUIRED = 3
	PIXEL_SUMMATION_PERCENT_DIFF = .007
	PIXEL_SUBTRACTION_PERCENT_DIFF = .02
	XOR_MATHCED_IMAGE_THRESHOLD = 96


	# The default constructor for your Agent. Make sure to execute any
	# processing necessary before your Agent starts solving problems here.
	#
	# Do not add any variables to this signature; they will not be used by
	# main().
	def __init__(self):
		self.here = sys.path[0]
		self.time = time.clock()
		self.max_transform_attempt_order = max([stat_trans.order for stat_trans in Trans.STATIC_TRANSFORMS])
		self.problem = None
		self.is3x3 = False

	# The primary method for solving incoming Raven's Progressive Matrices.
	# For each problem, your Agent's Solve() method will be called. At the
	# conclusion of Solve(), your Agent should return an int representing its
	# answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
	# are also the Names of the individual RavensFigures, obtained through
	# RavensFigure.getName(). Return a negative number to skip a problem.
	#
	# Make sure to return your answer *as an integer* at the end of Solve().
	# Returning your answer as a string may cause your program to crash.
	def Solve(self, problem):
		self.time = time.time()
		self.set_problem_details(problem)
		self.print_problem_details()

		# Load our images
		if self.is3x3:
			im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		else:
			im_a, im_b, im_c = self.load_problem_images()

		answer_guess = -1
		if self.is3x3:

			AND_answer = self.get_AND_answer()
			if AND_answer > -1:
				print("Solved with AND:", AND_answer)
				self.print_elapsed_time()
				return AND_answer

			OR_answer = self.get_OR_answer()
			if OR_answer > -1:
				print("Solved with OR:", OR_answer)
				self.print_elapsed_time()
				return OR_answer

			XOR_answer = self.get_XOR_answer()
			if XOR_answer > -1:
				print("Solved with XOR:", XOR_answer)
				self.print_elapsed_time()
				return XOR_answer

			AB_pixel_sub_answer = self.get_AB_pixel_subtraction_answer()
			if AB_pixel_sub_answer > -1:
				print('Solved with AB Pixel Subtraction:', AB_pixel_sub_answer)
				self.print_elapsed_time()
				return AB_pixel_sub_answer

			pixel_sum_answer = self.get_pixel_summation_answer()
			if pixel_sum_answer > -1:
				print("Solved with Pixel Summation:", pixel_sum_answer)
				self.print_elapsed_time()
				return pixel_sum_answer

			region_summation_answers = self.region_summation_answers()
			uniqueness_answers = self.unique_answers()
			# print('Region Answers:', region_summation_answers)
			# print('Uniqueness Answers:', uniqueness_answers)
			valid_answers = [a for a in region_summation_answers if a in uniqueness_answers]
			print('valid answers:', valid_answers)
			if len(valid_answers) == 1:
				print('Only one valid answer:', valid_answers[0])
				self.print_elapsed_time()
				return valid_answers[0]
			elif len(valid_answers) == 0:
				return -1

			progression_answer = self.get_progression_answer(valid_answers)
			if progression_answer > -1:
				print("Solved with Progression:", progression_answer)
				self.print_elapsed_time()
				return progression_answer

			print("Skipped")

		else:
			h_transforms, v_transforms = None, None

			transform_attempt_order = 0
			while answer_guess == -1 and transform_attempt_order <= self.max_transform_attempt_order:
				# Let's determine what horizontal and vertical transformations we have - iteratively
				h_transforms = self.get_priority_transforms(im_a, im_b, h_transforms, transform_attempt_order)
				v_transforms = self.get_priority_transforms(im_a, im_c, v_transforms, transform_attempt_order)
				transform_attempt_order += 1

				# Choose between horizontal and vertical transforms
				# Will choose the list that has the best match before additions and subtractions
				print()
				if h_transforms[0].score > v_transforms[0].score:
					print('Choosing horizontal transforms')
					transforms = h_transforms
					image_to_transform = im_c
				else:
					print('Choosing vertical transforms')
					transforms = v_transforms
					image_to_transform = im_b

				# Apply Transforms to get expected solutions
				solutions = [transform.apply_to(image_to_transform) for transform in transforms]

				# Test solutions for accuracy, returning the best one if it fits well enough
				answer_guess = self.find_matching_answer(solutions)

			if problem.name.endswith('08') and not self.submitting: self.print_solution_info(image_to_transform,
																							 transforms, solutions)

		self.print_elapsed_time()
		return answer_guess

	def load_problem_images(self):
		try:
			im_a = self.load_image('A')
			im_b = self.load_image('B')
			im_c = self.load_image('C')

			if self.is3x3:
				im_d = self.load_image('D')
				im_e = self.load_image('E')
				im_f = self.load_image('F')
				im_g = self.load_image('G')
				im_h = self.load_image('H')
				return Pillow.normalize(im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h)
			else:
				return Pillow.normalize(im_a, im_b, im_c)
		except IOError as e:
			print('IO issue - probably could not load image')
			print(e)

	def load_problem_answers(self):
		try:
			im1 = self.load_image('1')
			im2 = self.load_image('2')
			im3 = self.load_image('3')
			im4 = self.load_image('4')
			im5 = self.load_image('5')
			im6 = self.load_image('6')

			if self.is3x3:
				im7 = self.load_image('7')
				im8 = self.load_image('8')
				return Pillow.normalize(im1, im2, im3, im4, im5, im6, im7, im8)
			else:
				return Pillow.normalize(im1, im2, im3, im4, im5, im6)
		except IOError as e:
			print('IO issue - probably could not load image')
			print(e)

	# Limits answer options based on regions summation - returns valid answers to be considered
	def region_summation_answers(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		images = [im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h]
		image_region_counts = [Pillow.count_regions_dict(im) for im in images]

		row1_black_regions = sum([image_region_counts[i]['black'] for i in range(0, 3)])
		row1_white_regions = sum([image_region_counts[i]['white'] for i in range(0, 3)])
		row2_black_regions = sum([image_region_counts[i]['black'] for i in range(3, 6)])
		row2_white_regions = sum([image_region_counts[i]['white'] for i in range(3, 6)])

		limited_answers = []
		if row1_black_regions == row2_black_regions and row1_white_regions == row2_white_regions:
			print('Exhibits row-region summation')
			answer_region_counts = [Pillow.count_regions_dict(ans) for ans in answers]

			gh_black = image_region_counts[6]['black'] + image_region_counts[7]['black']
			gh_white = image_region_counts[6]['white'] + image_region_counts[7]['white']

			for i in range(len(answer_region_counts)):
				answer_dict = answer_region_counts[i]
				if row1_black_regions == gh_black + answer_dict['black'] and row1_white_regions == gh_white + answer_dict['white']:
					limited_answers.append(i+1)
					# print('{0} is valid'.format(i+1))

		return limited_answers if len(limited_answers) > 0 else [1, 2, 3, 4, 5, 6, 7, 8]

	# Limits answer options based on uniqueness
	def unique_answers(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		images = [im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h]
		answers = self.load_problem_answers()

		unique_answers = []
		if self.are_unique(images):
			print('Exhibits uniqueness')
			for i, answer in enumerate(answers):
				if self.is_unique(answer, images):
					unique_answers.append(i+1)

		return unique_answers if len(unique_answers) > 0 else [1, 2, 3, 4, 5, 6, 7, 8]

	# Returns true of false depending on whether all of the images are unique or not
	def are_unique(self, images):
		if len(images) == 0: return False

		for i, image in enumerate(images):
			if not self.is_unique(image, images[i+1:]): return False

		return True

	# Returns true or false depending on whether the first image is unique to all the other images
	def is_unique(self, im, images):
		for image in images:
			if Pillow.images_match(im, image): return False

		return True

	# Answer method that tests problem for "progression" pattern
	def get_progression_answer(self, valid_answers):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		# HORIZONTAL FIRST

		# Get our initial pixel analyses
		pa_row_1 = PixelAnalysis(im_a, im_b, im_c)
		pa_row_2 = PixelAnalysis(im_d, im_e, im_f)
		# print(pa_row_1); print(); print(pa_row_2); print()

		# Analyze those analyses
		paa = PixelAnalysisAnalysis(pa_row_1, pa_row_2)
		# print(paa); print()

		# Collect our pixel analyses for each answer and score them
		pa_answers = []
		for i, answer in enumerate(answers):
			if i+1 not in valid_answers: continue

			pa_answer = PixelAnalysis(im_g, im_h, answer)
			pa_answer.set_deviation(paa)
			pa_answer.answer = i+1
			# print('Answer {0}:'.format(i+1)); print(pa_answer); print()
			pa_answers.append(pa_answer)

		# Sort our answer pa's by score
		pa_answers.sort(key=lambda pa: pa.deviation)
		hor_best_pa_answer = pa_answers[0]
		hor_deviation_dif = pa_answers[1].deviation - pa_answers[0].deviation
		# print('Horizontal deviation is {0} for answer {1}. Best by {2}'.format(
		# 	pa_answers[0].deviation, pa_answers[0].answer, hor_deviation_dif
		# )); print()


		# VERTICAL SECOND

		# Get our initial pixel analyses
		pa_col_1 = PixelAnalysis(im_a, im_d, im_g)
		pa_col_2 = PixelAnalysis(im_b, im_e, im_h)
		# print(pa_row_1); print(); print(pa_row_2); print()

		# Analyze those analyses
		paa = PixelAnalysisAnalysis(pa_col_1, pa_col_2)
		# print(paa); print()

		# Collect our pixel analyses for each answer and score them
		pa_answers = []
		for i, answer in enumerate(answers):
			if i+1 not in valid_answers: continue

			pa_answer = PixelAnalysis(im_c, im_f, answer)
			pa_answer.set_deviation(paa)
			pa_answer.answer = i+1
			# print('Answer {0}:'.format(i+1)); print(pa_answer); print()
			pa_answers.append(pa_answer)

		# Sort our answer pa's by score
		pa_answers.sort(key=lambda pa: pa.deviation)
		ver_best_pa_answer = pa_answers[0]
		ver_deviation_dif = pa_answers[1].deviation - pa_answers[0].deviation
		# print('Vertical deviation is {0} for answer {1}. Best by {2}'.format(
		# 	pa_answers[0].deviation, pa_answers[0].answer, ver_deviation_dif
		# )); print()


		# Take the best from horizontal and vertical
		if hor_deviation_dif > ver_deviation_dif:
			chosen_deviation_dif = hor_deviation_dif
			chosen_pa_answer = hor_best_pa_answer
		else:
			chosen_deviation_dif = ver_deviation_dif
			chosen_pa_answer = ver_best_pa_answer

		if chosen_deviation_dif >= self.DEVIATION_DIFFERENCE_REQUIRED:
			return chosen_pa_answer.answer
		else:
			return -1

	# Answer method that tests problem for AND pattern
	def get_AND_answer(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		# If first and second row exhibit OR pattern, see if we can find an accurate third-row solution
		if self.exhibits_AND(im_a, im_b, im_c) and self.exhibits_AND(im_d, im_e, im_f):
			for i, answer in enumerate(answers):
				if self.exhibits_AND(im_g, im_h, answer):
					return i+1

		return -1

	# Helper method that returns true or false showing whether im1 ORed with im2 gives im3
	def exhibits_AND(self, im1, im2, im3):
		return Pillow.images_match(Pillow.AND_image(im1, im2), im3)

	# Answer method that tests problem for OR pattern, where A + B = C and/or A + D = G
	def get_OR_answer(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		# If first and second row exhibit OR pattern, see if we can find an accurate third-row solution
		if self.exhibits_OR(im_a, im_b, im_c) and self.exhibits_OR(im_d, im_e, im_f):
			for i, answer in enumerate(answers):
				if self.exhibits_OR(im_g, im_h, answer):
					return i+1

		return -1

	# Helper method that returns true or false showing whether im1 ORed with im2 gives im3
	def exhibits_OR(self, im1, im2, im3):
		return Pillow.images_match(Pillow.OR_image(im1, im2), im3)

	# Answer method that tests problem for XOR pattern
	def get_XOR_answer(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		# If first and second row exhibit XOR pattern, see if we can find an accurate third-row solution
		if self.exhibits_XOR(im_a, im_b, im_c) and self.exhibits_XOR(im_d, im_e, im_f):
			for i, answer in enumerate(answers):
				if self.exhibits_XOR(im_g, im_h, answer):
					return i+1

		# If first and second col exhibit XOR pattern, see if we can find an accurate third-col solution
		if self.exhibits_XOR(im_a, im_d, im_g) and self.exhibits_XOR(im_b, im_e, im_h):
			for i, answer in enumerate(answers):
				if self.exhibits_XOR(im_c, im_f, answer):
					return i+1

		return -1

	def exhibits_XOR(self, im1, im2, im3):
		return Pillow.get_image_match_score(Pillow.XOR_image(im1, im2), im3) >= self.XOR_MATHCED_IMAGE_THRESHOLD

	# Answer method that tests problem for total pixel summation pattern
	def get_pixel_summation_answer(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		# Try row summation first

		row_1_sum = Pillow.black_pixel_summation(im_a, im_b, im_c)
		row_2_sum = Pillow.black_pixel_summation(im_d, im_e, im_f)

		# If the sums are close enough, let's see if we can find an answer that is also close
		least_diff = 1000000
		best_answer = -1
		if abs(row_1_sum - row_2_sum) < self.PIXEL_SUMMATION_PERCENT_DIFF * min(row_1_sum, row_2_sum):
			avg = (row_1_sum + row_2_sum) // 2
			for i, answer in enumerate(answers):
				sum = Pillow.black_pixel_summation(im_g, im_h, answer)
				diff = abs(sum - avg)
				if diff < self.PIXEL_SUMMATION_PERCENT_DIFF * min(sum, avg) and diff < least_diff:
					least_diff = diff
					best_answer = i+1
					# print('new best answer of', i+1)

		# Try col summation second

		col_1_sum = Pillow.black_pixel_summation(im_a, im_d, im_g)
		col_2_sum = Pillow.black_pixel_summation(im_b, im_e, im_h)

		# If the sums are close enough, let's see if we can find an answer that is also close
		if abs(col_1_sum - col_2_sum) < self.PIXEL_SUMMATION_PERCENT_DIFF * min(col_1_sum, col_2_sum):
			avg = (col_1_sum + col_2_sum) // 2
			for i, answer in enumerate(answers):
				sum = Pillow.black_pixel_summation(im_c, im_f, answer)
				diff = abs(sum - avg)
				if diff < self.PIXEL_SUMMATION_PERCENT_DIFF * min(sum, avg) and diff < least_diff:
					least_diff = diff
					best_answer = i+1
					# print('new best answer of', i+1)

		return best_answer

	# Answer method that tests problem for A-B=C pixel summation pattern
	def get_AB_pixel_subtraction_answer(self):
		im_a, im_b, im_c, im_d, im_e, im_f, im_g, im_h = self.load_problem_images()
		answers = self.load_problem_answers()

		# Try row relationships first

		row_1_diff = abs((Pillow.count(im_a, 'black') - Pillow.count(im_b, 'black')) - Pillow.count(im_c, 'black'))
		row_2_diff = abs((Pillow.count(im_d, 'black') - Pillow.count(im_e, 'black')) - Pillow.count(im_f, 'black'))
		# print('Row 1 and 2 diff:', row_1_diff, row_2_diff)
		# print('C and F count:', Pillow.count(im_c, 'black'), Pillow.count(im_f, 'black'))

		# If the subtractions are close enough, let's see if we can find an answer that is also close
		least_diff = 10000000
		best_answer = -1
		if row_1_diff < self.PIXEL_SUBTRACTION_PERCENT_DIFF * Pillow.count(im_c, 'black') and row_2_diff < self.PIXEL_SUBTRACTION_PERCENT_DIFF * Pillow.count(im_f, 'black'):
			gh_diff = Pillow.count(im_g, 'black') - Pillow.count(im_h, 'black')
			for i, answer in enumerate(answers):
				diff = abs(gh_diff - Pillow.count(answer, 'black'))
				if diff < self.PIXEL_SUBTRACTION_PERCENT_DIFF * Pillow.count(answer, 'black') and diff < least_diff:
					least_diff = diff
					best_answer = i+1

		# Try col summation second

		col_1_diff = abs((Pillow.count(im_a, 'black') - Pillow.count(im_d, 'black')) - Pillow.count(im_g, 'black'))
		col_2_diff = abs((Pillow.count(im_b, 'black') - Pillow.count(im_e, 'black')) - Pillow.count(im_h, 'black'))

		# If the subtractions are close enough, let's see if we can find an answer that is also close
		if col_1_diff < self.PIXEL_SUBTRACTION_PERCENT_DIFF * Pillow.count(im_g, 'black') and col_2_diff < self.PIXEL_SUBTRACTION_PERCENT_DIFF * Pillow.count(im_h, 'black'):
			cf_diff = Pillow.count(im_c, 'black') - Pillow.count(im_f, 'black')
			for i, answer in enumerate(answers):
				diff = abs(cf_diff - Pillow.count(answer, 'black'))
				if diff < self.PIXEL_SUBTRACTION_PERCENT_DIFF * Pillow.count(answer, 'black') and diff < least_diff:
					least_diff = diff
					best_answer = i+1

		return best_answer

	# Given two images, returns a list of Transforms that will turn im1 into im2
	# List is ordered by how well the images matched before additions and subtractions were considered: best match first
	def get_priority_transforms(self, im1, im2, transforms=[], order=0):
		priority_transforms = [] if transforms is None else transforms

		if len(priority_transforms) == 0:
			priority_transforms = [Trans.Transform(im1)]  # Start the list with a blank transform

		# For each static transform, add a Transform to the list
		# Only add transforms of the current order - iterative solution
		for stat_trans in filter(lambda t: t.order == order, Trans.STATIC_TRANSFORMS):
			priority_transforms.append(Trans.Transform(im1).add_static_transform(stat_trans))

		# Score each transform for ordering
		for transform in filter(lambda t: t.score is None, priority_transforms):
			transform.score = Pillow.get_image_match_score(transform.current_image, im2)
		# Order our list by how well each one matches im2
		priority_transforms.sort(key=lambda t: t.score, reverse=True)

		# Put in the add and subtract images
		for transform in filter(lambda t: t.add_image is None or t.subtract_image is None, priority_transforms):
			if not Pillow.images_match(transform.current_image, im2):
				transform.set_additions(im2)
				transform.set_subtractions(im2)

		return priority_transforms

	# Given the problem and a list of expected solutions, tests the solutions against the
	# 	provided answers in the problem to find the best match
	# Returns the number representing the chosen answer, the return for the Agent's Solve method
	def find_matching_answer(self, solution_images):
		answers = self.load_problem_answers()

		# Get the best match from the answers for each solution image
		solutions = []
		for solution_image in solution_images:
			chosen_answer = 0
			percent_match = 0
			for i in range(len(answers)):
				answer = answers[i]

				match_score = Pillow.get_image_match_score(solution_image, answer, fuzzy=True)
				if match_score > percent_match:
					percent_match = match_score
					chosen_answer = i + 1

			solutions.append(Solution(chosen_answer, percent_match))

		# print('Solution Scores:', [str(s) for s in solutions])

		# Pick the best solution (with the highest answer match percentage)
		solutions.sort(key=lambda s: s.percent_match, reverse=True)
		chosen_solution = solutions[0]

		print('Chosen Solution is ', chosen_solution)

		if chosen_solution.percent_match < Pillow.MATCHED_IMAGE_THRESHOLD:
			print('No decent match. Giving up.')
			return -1

		return chosen_solution.answer

	def set_problem_details(self, problem):
		self.problem = problem
		self.is3x3 = self.problem.problemType == '3x3'

	def print_problem_details(self):
		print()
		print('==================================================')
		print('About to solve:', self.problem.name, '(' + self.problem.problemType + ')')
		print('==================================================')

	def print_elapsed_time(self):
		elapsed = time.time() - self.time
		print()
		print('Solution took', int(elapsed * 1000), 'milliseconds')
		self.time = time.time()

	# Returns the image with the same name as the given key
	def load_image(self, key):
		filename = self.problem.figures[key].visualFilename
		return Pillow.Image.open(os.path.join(self.here, filename))

	def print_solution_info(self, start_image, transforms, solution_images):
		start_image.save(os.path.join(self.here, 'testAgent', 'startImage.png'))

		print('  Printing Solution Info:')
		for i in range(len(transforms)):
			transform = transforms[i]
			solution = solution_images[i]
			print('   ', [t.type for t in transform.static_transforms])
			print('    Score:', transform.score)
			print('    Added:', transform.add_percent)
			print('    Subtracted:', transform.subtract_percent)
			print()
			solution.save(os.path.join(self.here, 'testAgent', 'solution{0!s}.png'.format(i + 1)))


class Solution:
	def __init__(self, answer, percent_match):
		self.answer = answer
		self.percent_match = percent_match

	def __str__(self):
		return '{0!s}: {1!s}%'.format(self.answer, self.percent_match)


# To measure the change in the images
class PixelAnalysis:

	MIN_PIXEL_CHANGE_RANGE = 1
	MIN_MATCH_RATE_CHANGE_RANGE = .1


	def __init__(self, *images):
		self._images = list(images)
		self.pixel_changes = []
		self.pixel_change_dif = None  # Average difference between the pixel changes
		self.match_rates = []
		self.match_rate_change_dif = None  # Average difference between the match rate changes
		self.deviation = None
		self.answer = None

		self._set_pixel_changes()
		self._set_pixel_change_dif()
		self._set_match_rates()
		self._set_match_rate_change_dif()

	def _set_pixel_changes(self):
		for i in range(len(self._images)):
			if i + 1 > len(self._images) - 1: break

			im1 = self._images[i]
			im2 = self._images[i + 1]
			self.pixel_changes.append(Pillow.black_pixel_count_difference(im1, im2))

	def _set_pixel_change_dif(self):
		difs = []
		for i in range(len(self.pixel_changes)):
			if i + 1 > len(self.pixel_changes) - 1: break

			change1 = self.pixel_changes[i]
			change2 = self.pixel_changes[i + 1]
			difs.append(change2 - change1)

		self.pixel_change_dif = sum(difs) // len(difs)  # Average difference

	def _set_match_rates(self):
		for i in range(len(self._images)):
			if i + 1 > len(self._images) - 1: break

			im1 = self._images[i]
			im2 = self._images[i+1]
			self.match_rates.append(Pillow.black_match_rate(im1, im2))

	def _set_match_rate_change_dif(self):
		difs = []
		for i in range(len(self.match_rates)):
			if i + 1 > len(self.match_rates) - 1: break

			rate1 = self.match_rates[i]
			rate2 = self.match_rates[i + 1]
			difs.append(rate2 - rate1)

		self.match_rate_change_dif = sum(difs) / len(difs)

	def set_deviation(self, paa):
		pixel_change_range_size = abs(paa.expected_pixel_dif_range[0] - paa.expected_pixel_dif_range[1])
		match_rate_change_range_size = abs(paa.expected_match_rate_dif_range[0] - paa.expected_match_rate_dif_range[1])

		# Make sure our ranges are not zero
		if pixel_change_range_size == 0: pixel_change_range_size = self.MIN_PIXEL_CHANGE_RANGE
		if match_rate_change_range_size == 0: match_rate_change_range_size = self.MIN_MATCH_RATE_CHANGE_RANGE

		# Pixel change dif deviation first
		if self.pixel_change_dif > paa.expected_pixel_dif_range[1]:  # If we got more than expected
			raw_deviation = self.pixel_change_dif - paa.expected_pixel_dif_range[1]
			pixel_change_dif_deviation = math.ceil(raw_deviation / pixel_change_range_size)
		elif self.pixel_change_dif < paa.expected_pixel_dif_range[0]:  # If we got less than expected
			raw_deviation = paa.expected_pixel_dif_range[0] - self.pixel_change_dif
			pixel_change_dif_deviation = math.ceil(raw_deviation / pixel_change_range_size)
		else:
			pixel_change_dif_deviation = 0  # Within the expected

		# Match rate change dif deviation second
		if self.match_rate_change_dif > paa.expected_match_rate_dif_range[1]:  # If we got more than expected
			raw_deviation = self.match_rate_change_dif - paa.expected_match_rate_dif_range[1]
			match_rate_change_dif_deviation = math.ceil(raw_deviation / match_rate_change_range_size)
		elif self.match_rate_change_dif < paa.expected_match_rate_dif_range[0]:  # If we got less than expected
			raw_deviation = paa.expected_match_rate_dif_range[0] - self.match_rate_change_dif
			match_rate_change_dif_deviation = math.ceil(raw_deviation / match_rate_change_range_size)
		else:
			match_rate_change_dif_deviation = 0  # Within the expected

		self.deviation = pixel_change_dif_deviation + match_rate_change_dif_deviation

	def __str__(self):
		return (
			str(self.pixel_changes) + '\n' +
			str(self.pixel_change_dif) + '\n' +
			str(self.match_rates) + '\n' +
			str(self.match_rate_change_dif) +
			('' if self.deviation is None else '\nDeviation: {0}'.format(self.deviation))
		)


# To measure the difference in the change
class PixelAnalysisAnalysis:
	def __init__(self, pa1, pa2):
		self.pa1, self.pa2 = pa1, pa2
		self.pixel_dif_change = None
		self.expected_pixel_dif = None
		self.expected_pixel_dif_range = None
		self.match_rate_dif_change = None
		self.expected_match_rate_dif = None
		self.expected_match_rate_dif_range = None

		self._set_pixel_dif_change()
		self._set_match_rate_dif_change()

	def _set_pixel_dif_change(self):
		self.pixel_dif_change = self.pa2.pixel_change_dif - self.pa1.pixel_change_dif
		self.expected_pixel_dif = self.pa2.pixel_change_dif + self.pixel_dif_change
		self.expected_pixel_dif_range = (
			min(self.pa1.pixel_change_dif, self.pa2.pixel_change_dif, self.expected_pixel_dif),
			max(self.pa1.pixel_change_dif, self.pa2.pixel_change_dif, self.expected_pixel_dif)
		)

	def _set_match_rate_dif_change(self):
		self.match_rate_dif_change = self.pa2.match_rate_change_dif - self.pa1.match_rate_change_dif
		self.expected_match_rate_dif = self.pa2.match_rate_change_dif + self.match_rate_dif_change
		self.expected_match_rate_dif_range = (
			min(self.pa1.match_rate_change_dif, self.pa2.match_rate_change_dif, self.expected_match_rate_dif),
			max(self.pa1.match_rate_change_dif, self.pa2.match_rate_change_dif, self.expected_match_rate_dif)
		)

	def __str__(self):
		return (
			str(self.expected_pixel_dif_range) + '\n'
			+ str(self.expected_match_rate_dif_range)
		)