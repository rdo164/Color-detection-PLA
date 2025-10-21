import cv2
import os
import numpy as np
import clase as cls # to avoid name conflict with 'class' keyword
from typing import Any, List, Tuple
from imutils.video import VideoStream
import hue_histogram as hh


def create_window(name: str, img: Any, w: int, h: int, x: int, y: int) -> None:
    """
    Creates a positioned, fixed-size window and displays the image.
    Args:
        name: name of the window (string).
        img: image of type numpy.ndarray (BGR) or any object accepted by cv2.imshow.
        w: width of the window in pixels.
        h: height of the window in pixels.
        x: x-coordinate of the window (origin on the screen).
        y: y-coordinate of the window.
    """
    
    if img is None:
        raise ValueError(f"Imagen vacía para ventana '{name}'")

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # allows resize
    cv2.resizeWindow(name, w, h)               # desire size 
    cv2.moveWindow(name, x, y)                 # position
    cv2.imshow(name, img)           


def pre_process(src: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-processing: converts from BGR to HSV and separates the H, S, and V channels.

    Args:
    src: input image (BGR) as a numpy.ndarray.

    Returns:
    H, S, V: single-channel arrays (numpy.ndarray).

    Raises:
    ValueError: if src is None or empty.
    """

    # if src is None or src.size == 0:
    #     raise ValueError("Imagen de entrada vacía")

    # In case the image is grayscale, convert to BGR to avoid errors
    #if src.ndim == 2 or (src.ndim == 3 and src.shape[2] == 1):
    #    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    return H, S, V

def segmentacion(src: np.ndarray, H: np.ndarray, S: np.ndarray, V: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmentación: máscara + morfología + foreground.

    Args:
        src: imagen BGR (uint8).
        H, S, V: canales HSV individuales (uint8).

    Returns:
        result: imagen BGR con sólo el foreground (src & mask).
        obj_mask: máscara binaria uint8 (0 o 255).
    """
    if src is None or src.size == 0:
        raise ValueError("src vacío")
    if V is None or V.size == 0 or S is None or S.size == 0:
        raise ValueError("Canales S/V inválidos")

    # 3.1) Umbral de Otsu sobre V (objetos brillantes)
    # Umbral of OTSU on V (bright objects)
    # cv2.thresold returnrs (ret, dst)
    # make sure that V is uint8
    V_u8 = V.astype(np.uint8)
    _, obj_mask = cv2.threshold(V_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3.2 ) Keep only pixels with medium/high saturation (S >= 20)
    # If S is single-channel, inRange works with scale 0..255
    sat_keep = cv2.inRange(S, 20, 255)
    obj_mask = cv2.bitwise_and(obj_mask, sat_keep)


    # 3.3) Morpholofic cleaning
    e_k1 = (9, 9)
    e_k2 = (7, 7)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, e_k1)  # OPEN: quits noise
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, e_k2)  # CLOSED: fill the gaps
    

    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, k1, iterations=1)   # deletes the white dots noise dots
    ### !!!!! REVIEW THIS PART maybe the problem for white objects
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, k2, iterations=1)  # fill black gaps inside white objects

    # 3.4) Apply mask to original (colored foreground)
    # cv2.bitwise_and accepts the mask= argument for single-channel masks
    result = cv2.bitwise_and(src, src, mask=obj_mask)

    return e_k1, e_k2 ,result, obj_mask


def feature_extraction(k1,k2,src: np.ndarray, obj_mask: np.ndarray, result: np.ndarray,
                       area_thresh: float = 20000.0) -> Tuple[int, List[np.ndarray]]:
    """
    Feature extraction: contours + color means.

    Args:
        src: original BGR image.
        obj_mask: binary mask (uint8) with objects (0/255).
        result: BGR image where the mask was applied (foreground).
        area_thresh: minimum area to consider an object as a piece (default 20000).

    Returns:
        (pieces_count, pieces_imgs)
        - pieces_count: number of large pieces found.
        - pieces_imgs: list of colored (BGR) cropped images of each large piece.
    """

    if src is None or src.size == 0:
        raise ValueError("src vacío")
    if obj_mask is None or obj_mask.size == 0:
        raise ValueError("obj_mask vacío")
    if result is None or result.size == 0:
        raise ValueError("result vacío")

    # !!!! Take a look findContours en Python puede devolver 2 o 3 valores según la versión
    # 5.1) findcontours in the image
    contours_info = cv2.findContours(obj_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5.2) Unpack contours depending on OpenCV version
    if len(contours_info) == 3:

        # hierachy indicates how contours are related between them  
        _, contours, hierarchy = contours_info
    else:
        # hierachy indicates how contours are related between them  
        contours, hierarchy = contours_info

    print(f"Cantidad de contornos (todos): {len(contours)}")

    piezas_count = 0
    piezas_imgs: List[np.ndarray] = []
    
    # 5.3) For each contour, compute area, mean color, and extract colored cutout
    for i, cnt in enumerate(contours):

        area = cv2.contourArea(cnt)
        # Create an instance of Cluster for each contour 
        piece = cls.Cluster()

        # set properties
        piece.area = area
        piece.morph_size = k1
        piece.morph_size2 = k2

        # 5.4) validate the area of the contour
        if area > area_thresh:
            piezas_count += 1

            # i object mask  
            mask = np.zeros(result.shape[:2], dtype=np.uint8)

            cv2.drawContours(mask, contours, i, color=255, thickness=cv2.FILLED)
            
            # Obtain colors HSV            
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv, mask=mask)

            print(f"Objeto {i} - color medio (HSV): {mean_hsv[:3]}")

            mean_h, mean_s, mean_v = map(int, mean_hsv[:3])
            piece.h_value = mean_h
            piece.s_value = mean_s      
            piece.v_value = mean_v

            # colored cutout of the object
            obj_pixels = cv2.bitwise_and(result, result, mask=mask)
            piezas_imgs.append(obj_pixels)

            # media color BGR (cv2.mean returns (b,g,r,a))
            mean_color = cv2.mean(result, mask=mask)
            mean_bgr = mean_color[:3]
            print(f"Objeto {i} - area: {area} - color medio (BGR): {mean_bgr}")

    # Contours Visualization againts the original
    vis = src.copy()
    if len(contours) > 0:
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    try:
        create_window("Contours", vis, 500, 300, 700, 0)
    except NameError:
        # Si no está definida, mostramos con cv2.imshow como fallback
        cv2.imshow("Contours", vis)

    # Show big pieces found
    for idx, p_img in enumerate(piezas_imgs):
        window_x = 100 + idx * 350
        try:
            create_window(f"Pieza {idx}", p_img, 300, 300, window_x, 400)
            
            hist = hh.create_hue_histogram(p_img)
            hh.plot_hue_histogram(hist)  
        except NameError:
            cv2.imshow(f"Pieza {idx}", p_img)

    return piezas_count, piezas_imgs


def run_pipeline(img_path):
    # 1) Cargue image
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)


    # 2) Pre-process
    H, S, V = pre_process(src)  # Assuming pre_process returns H, S, V

    # 3) Segmentation
    k1,k2,result, obj_mask = segmentacion(src, H, S, V)  # Assuming segmentacion returns result, objMask

    # 4) Visualize results
    #create_window("Original",   src,      500, 300,   0,   0)
    #create_window("Mask",       obj_mask, 500, 300, 100,   0)
    #create_window("Foreground", result,   500, 300, 300,   0)

    # 5) Features
    feature_extraction(k1,k2,src, obj_mask, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    run_pipeline("./img/IMG_8819.JPEG")
