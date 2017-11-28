import imageio

left = 3800
right = 4000
filenames = []
images = []
for i in range(left, right+1):
    filenames.append('img_' + str(i) + '.png')
for f in filenames:
    images.append(imageio.imread(f))
imageio.mimsave('img_stay_top.gif', images)
