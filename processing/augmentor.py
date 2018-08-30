import Augmentor

p = Augmentor.Pipeline("./input")
p.flip_random(probability=0.5)
p.skew(probability=0.5)
# p.random_distortion(probability=0.3, grid_width=5, grid_height=5, magnitude=2)
p.shear(probability=0.6, max_shear_left=15, max_shear_right=15)
p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
p.crop_random(probability=1.0, percentage_area=0.7)
p.crop_by_size(probability=1.0, width=80, height=80)
p.sample(260)