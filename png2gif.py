import imageio

left = 45
right = 120
filenames = []
images = []
for i in range(left, right+1):
    filenames.append('img_' + str(i) + '.png')
for f in filenames:
    images.append(imageio.imread(f))
imageio.mimsave('img_v0_good_2.gif', images)