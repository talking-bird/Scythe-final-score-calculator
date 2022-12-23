
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale
from matplotlib import cm
from skimage.transform import ProjectiveTransform, warp
from skimage.measure import LineModelND, ransac
from skimage.feature import match_template



def show(img, cmap=None):
  plt.imshow(img) if cmap is None else plt.imshow(img, cmap=cmap)
  plt.show()

def preprocess_img(img):
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  img = np.uint8(rescale(img, (0.25,0.25,1))*255)
  return img

# ## Cut the board from the image

def transform(img, visual=False):
  plt.imshow(img)
  plt.show()
  # show(img)
  gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  blurred = cv.GaussianBlur(gray, ksize=(7,7),sigmaX=0,sigmaY=0)
  edges = cv.Canny(blurred,20,150)
  if visual:
    show(edges,'gray')

  borders_hough = np.ones_like(img, dtype=np.uint8)*255
  lines = cv.HoughLines(edges,1,np.pi/180,200)
  lines = list(lines)
  lines = list(filter(lambda x: x[0,1] !=0, lines))
  lines = np.concatenate(lines, axis=0)
  # convert all lines to a-b space
  ab_selected = np.array(
      [hough_thetaro_to_ab(theta, ro)
      for ro, theta in lines]
  )
    
  plt.figure(figsize = (8,6))
  plt.imshow(borders_hough)

  y_max, x_max = borders_hough.shape[:-1]
  for line in lines:
      ro, theta = line
      a, b = hough_thetaro_to_ab(theta, ro)
      plt.plot(*get_line(a, b, [0, x_max]), 'b', lw=2)
  plt.show()
  
  # look for inliers using a linear model
  _ , inliers = ransac(ab_selected, LineModelND, min_samples=2, residual_threshold=0.01)



  # points for transform
  lines_ab_final = ab_selected[inliers]

  points_current = []
  points_desired = []

  for i, ab in enumerate(lines_ab_final):
    # the points for the left of the image (x=0)
    # y = a*x+b
    y_0 = ab[1]
    points_current.append([0, y_0])
    points_desired.append([0, y_0])
    # the points for the right of the image (x=x_max)
    points_current.append([x_max, ab[0]*x_max+ab[1]])
    points_desired.append([x_max, y_0])

  points_current = np.array(points_current)
  points_desired = np.array(points_desired)
  #transform
  tform = ProjectiveTransform()
  tform.estimate(points_desired, points_current)
  image_warped = warp(img, tform)

  y_borders = int(np.min(points_desired[:,1])), int(np.max(points_desired[:,1]))
  image_warped_cut = image_warped[y_borders[0]:y_borders[1]]
  return np.uint8(image_warped_cut*255)

# two auxilary functions
def get_line(a, b, x_lim, y_lim=None):
    x = np.linspace(*x_lim)
    y = a * x + b
    if y_lim is not None:
        y = y[y>=y_lim[0]]
        y = y[y<=y_lim[1]]
    return x, y

def hough_thetaro_to_ab(theta, ro):
    # rho = x*cos(theta)+y*sin(theta)
    # y = a*x+b
    if theta == 0:
      theta =1e-4
    a = -1 /np.tan(theta)
    b = ro / np.sin(theta)
    return a, b

# ## Color sigmentation for stars


def mask_coins_hsv(img, img_hsv,hue_min,hue_max,lit_min,lit_max,sat_min,sat_max):
  mask = cv.inRange(img_hsv, (hue_min,lit_min,sat_min),(hue_max,lit_max,sat_max))
  result = cv.bitwise_and(img, img, mask=mask)
  return result


def mask_coins_rgb(img, img_rgb, r_min,r_max,g_min,g_max,b_min,b_max):
  mask = cv.inRange(img, (r_min,g_min,b_min),(r_max,g_max,b_max))
  result = cv.bitwise_and(img, img, mask=mask)
  return result

def get_masks(img, img_hsv):

    hue_min,hue_max,lit_min,lit_max,sat_min,sat_max = \
      90, 172, 113, 202, 83, 255
    blue = mask_coins_hsv(img, img_hsv, hue_min,hue_max,lit_min,lit_max,sat_min,sat_max)

    hue_min,hue_max,lit_min,lit_max,sat_min,sat_max = \
      150, 200, 100, 200, 83, 190
    red = mask_coins_hsv(img, img_hsv, hue_min,hue_max,lit_min,lit_max,sat_min,sat_max)

    hue_min,hue_max,lit_min,lit_max,sat_min,sat_max = \
      0, 50, 90, 200, 195, 255
    yellow = mask_coins_hsv(img, img_hsv, hue_min,hue_max,lit_min,lit_max,sat_min,sat_max);

    r_min,r_max,g_min,g_max,b_min,b_max = \
      49,91,49,167,45,115
    black = mask_coins_rgb(img, img, r_min,r_max,g_min,g_max,b_min,b_max);
    return [blue,red,yellow,black]


def is_bound_box_ratio_good(contour):
    cv.boxPoints(cv.minAreaRect(contour))
    _,_,w,h = cv.boundingRect(contour)
    return h !=0 and w !=0 and 0.8<w/h<1.3

def get_star_numbers(img, visual=True):
  img = preprocess_img(img)
  img = transform(img)
  img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
  if visual:
      show(img)
  N_stars = {}
  masks = get_masks(img, img_hsv)
  masks_names = ['blue','red','yellow','black']
  for color, colorname in zip(masks,masks_names):
    gray = cv.cvtColor(color, cv.COLOR_RGB2GRAY)
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, np.ones((7,7)))
    blurred = cv.GaussianBlur(gray, ksize=(5,5),sigmaX=0,sigmaY=0)
    edges = cv.Canny(blurred, 15, 120)

    if visual:
      show(edges,'gray')
    if colorname in ['blue', 'red', 'yellow']:
      contours, hierarchy = cv.findContours(edges,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      cont_areas =[]
      img_cont = img.copy()
      contours = list(filter(lambda x: cv.contourArea(x)>500,contours))
      contours = list(filter(lambda x: cv.contourArea(x)<3000,contours))
      contours = list(filter(is_bound_box_ratio_good,contours))

      for contour in contours:
        cont_area = cv.contourArea(contour)
        cont_areas.append(cont_area)
      cv.drawContours(img_cont, contours, -1, (0, 255,0),10)
      if visual:
        show(img_cont)
      mean_area = np.mean(cont_areas)
      min_area = np.min(cont_areas)
      max_area = np.max(cont_areas)

      print(f'{colorname}: n_contours={len(contours)}, {mean_area=:5.2f}, {max_area=}, {min_area=}')
      N_stars[colorname] = len(contours)
    else:
      template = np.load(f'stars/templates/{colorname}.npy')
      corr_skimage = match_template(gray, template, pad_input=True)
      matches = corr_skimage>0.6
      show(np.uint8(matches*255), 'gray')

      contours, hierarchy = cv.findContours(np.uint8(matches*255),cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
      print(len(contours))
      N_stars[colorname] = len(contours)

      
  return N_stars

if __name__ == '__main__':
  N_stars_truth = {#'stars_empty1.jpg': {'blue':0,'red':0,'yellow':0,'black':0},
                  #  'stars_empty2.jpg': {'blue':0,'red':0,'yellow':0,'black':0},
                  #  'stars_empty3.jpg': {'blue':0,'red':0,'yellow':0,'black':0},
                   'stars_1.jpg': {'blue':6,'red':4,'yellow':4,'black':4},
                   'stars_2.jpg': {'blue':6,'red':4,'yellow':4,'black':4},
                   'stars_3.jpg': {'blue':6,'red':4,'yellow':4,'black':4},
                   }
  folder = 'stars'
  for file in N_stars_truth.keys():
    print(file)
    img = cv.imread(f'{folder}/{file}')
    N_stars = get_star_numbers(img, visual=False)
    print(N_stars)
    total_error = 0
    for colorname in ['blue','red','yellow','black']:
      error = abs(N_stars[colorname] - N_stars_truth[file][colorname])
      total_error += error
    print(f'{total_error = }')

# black_template = black_[120:210,670:770]
# if visual:
  # show(black_template)

# 

# 
# black_gray = cv.cvtColor(black_, cv.COLOR_RGB2GRAY)
# black_template_gray = cv.cvtColor(black_template, cv.COLOR_RGB2GRAY)
# show(black_gray,'gray')
# # show(black_template_gray,'gray')
# corr_skimage = match_template(black_gray, black_template_gray, pad_input=True)
# matches = corr_skimage>0.7
# show(np.uint8(matches*255), 'gray')

# # edges = cv.Canny(np.uint8(matches*255), 20, 150)
# # show(edges)
# contours, hierarchy = cv.findContours(np.uint8(matches*255),cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# print(len(contours))


