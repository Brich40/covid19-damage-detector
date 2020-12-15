import cv2
import numpy as np
import imutils


class ImageProcess:

    @staticmethod
    def color_seg(choice, path):
        if choice == 'white':
            lower_hue = np.array([0, 0, 0])
            upper_hue = np.array([0, 0, 255])
        elif choice == 'black':
            lower_hue = np.array([0, 0, 0])
            upper_hue = np.array([120, 120, 120])
        return lower_hue, upper_hue

    @staticmethod
    def image_edge(path_in, path_out):
        # Take each frame
        frame = cv2.imread(path_in)
        # frame = cv2.imread('images/road_1.jpg')

        frame = imutils.resize(frame, height=300)
        chosen_color = 'black'

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of a color in HSV
        lower_hue, upper_hue = ImageProcess.color_seg(chosen_color, path_in)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_hue, upper_hue)

        result_path = path_out + "stage_2.png"
        cv2.imwrite(result_path, mask)
        return result_path

    @staticmethod
    def convert_black_to_transparent(input_path, output_path):
        # Input image
        input = cv2.imread(input_path, cv2.IMREAD_COLOR)

        # Convert to RGB with alpha channel
        output = cv2.cvtColor(input, cv2.COLOR_BGR2RGBA)

        # Color to make transparent
        col = (0, 0, 0)

        # Color tolerance
        tol = (220, 220, 220)

        # Temporary array (subtract color)
        temp = np.subtract(input, col)

        # Tolerance mask
        mask = (np.abs(temp) <= tol)
        mask = (mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])

        # Generate alpha channel
        temp[temp < 0] = 0  # Remove negative values
        alpha = (temp[:, :, 0] + temp[:, :, 1] + temp[:, :, 2]) / 3  # Generate mean gradient over all channels
        alpha[mask] = alpha[mask] / np.max(alpha[mask]) * 255  # Gradual transparency within tolerance mask
        alpha[~mask] = 255  # No transparency outside tolerance mask

        # Set alpha channel in output
        output[:, :, 3] = alpha

        # Output images
        cv2.imwrite(output_path + 'input.png', alpha)
        result_path = output_path + 'stage_1.png'
        cv2.imwrite(result_path, output)
        return result_path

    @staticmethod
    def get_lung_contours(file_in, file_out):
        # == Parameters =======================================================================
        BLUR = 15
        lung_info = []

        # -- Read image -----------------------------------------------------------------------
        img = cv2.imread(file_in)

        # -- Find contours in edges, sort by area ---------------------------------------------
        contour_info = []

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        dimensions = img.shape
        for index, contour in enumerate(contour_info[0:2]):
            contour_center = ImageProcess.get_contour_position(contour[0])
            lung_side = "left" if contour_center["x"] < dimensions[1]/2 else "right"

            mask = np.zeros(img.shape)
            cv2.fillConvexPoly(mask, contour[0], (255))

            # -- Smooth mask, then blur it --------------------------------------------------------
            mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

            # -- Blend masked img into MASK_COLOR background --------------------------------------
            img = img.astype('float32') / 255.0  # for easy blending

            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.imwrite(file_out + "stage_4.png", img * 255)
            lung_image_path = file_out + "stage_5_" + lung_side + ".png"
            cv2.imwrite(lung_image_path, mask)

            lung_info.append({
                "image_path": lung_image_path,
                "side": lung_side,
                "area": contour[2],
                "x_center": contour_center["x"],
                "y_center": contour_center["y"],
            })
        return lung_info

    @staticmethod
    def get_contour_position(c):
        m = cv2.moments(c)
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        return {"x": x, "y": y}

    @staticmethod
    def add_lung_damage(lungs_info):
        lungs_info_with_damage = []

        normal_left_lung_area = 12000
        normal_right_lung_area = 10500

        for lung_info in lungs_info:
            if lung_info['side'] == 'left':
                lung_info['damage_percent'] = "%.2f" % (100 - (lung_info['area']/normal_left_lung_area) * 100)
            elif lung_info['side'] == 'right':
                lung_info['damage_percent'] = "%.2f" % (100 - (lung_info['area'] / normal_right_lung_area) * 100)
            if float(lung_info['damage_percent']) < 0 or float(lung_info['damage_percent'])>100:
                lung_info['damage_percent'] = "0.00"
            lungs_info_with_damage.append(lung_info)

        return lungs_info_with_damage

    @staticmethod
    def add_damage_percent(lung_info):
        image = cv2.imread(lung_info['image_path'])
        cv2.putText(image, "dommage : ",
                    (lung_info['x_center'] - 60, lung_info['y_center'] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 16, 238), 2)

        cv2.putText(image, str(lung_info['damage_percent']) + "%",
                    (lung_info['x_center'] - 45, lung_info['y_center']),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 16, 238), 2)
        cv2.imwrite(lung_info['image_path'], image)
