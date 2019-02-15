import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
import csv
from isg_ai_pb2 import ImageYoloBoxesPair
import shutil
import lmdb


# Define the inputs
# ****************************************************

output_filepath = '/scratch/Argus/cnn-data/'

image_filepath = '/scratch/Argus/cnn-data/All_Slides_Tiled_into_8k_Blocks/data/'
csv_filepath = '/scratch/Argus/cnn-data/All_Slides_Tiled_into_8k_Blocks/csv/'

# image_filepath = '/scratch/Argus/cnn-data/All_Slides_Tiled_into_8k_Blocks/simulated-images-1M/'
# csv_filepath = '/scratch/Argus/cnn-data/All_Slides_Tiled_into_8k_Blocks/simulated-images-1M/'


# size of the output images
tile_size = 512 # to allow for random crops down to 512 tiles
# ****************************************************


def load_saved_positions(filepath):
    A = []
    class_label = 0 # there is only one class so far in the star data

    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                vec = []
                vec.append(int(row['X']))
                vec.append(int(row['Y']))
                vec.append(int(row['Width']))
                vec.append(int(row['Height']))
                vec[2] = vec[0]+vec[2]-1 # convert from width to x_end
                vec[3] = vec[1]+vec[3]-1 # convert from height to y_end
                vec.append(int(class_label))
                A.append(vec)

    # [left, top, right, bottom]
    A = np.asarray(A, dtype=np.float)
    return A


def read_image(fp):
    # img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    img = skimage.io.imread(fp, as_grey=True)
    return img


def compute_intersection(box, boxes):
    # all boxes are [left, top, right, bottom]

    intersection = 0
    if boxes.shape[0] > 0:
        # this is the iou of the box against all other boxes
        x_left = np.maximum(box[0], boxes[:, 0])
        y_top = np.maximum(box[1], boxes[:, 1])
        x_right = np.minimum(box[2], boxes[:, 2])
        y_bottom = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(y_bottom - y_top, 0) * np.maximum(x_right - x_left, 0)

    return intersection


def write_img_to_db(txn, img, boxes, key_str):
    # label 0 is background, 1 is foreground
    img = np.asarray(img, dtype=np.uint8)
    boxes = np.asarray(boxes, dtype=np.int32)

    datum = ImageYoloBoxesPair()
    datum.channels = 1
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]
    datum.image = img.tobytes()
    datum.box_count = boxes.shape[0]
    if boxes.shape[0] > 0:
        datum.boxes = boxes.tobytes()

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


def process_slide_tiling(img, bbStar):
    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]
    delta = int(tile_size - 32)

    img_list = []
    box_list = []

    for x_st in range(0, (width - tile_size), delta):
        for y_st in range(0, (height - tile_size), delta):
            x_end = x_st + tile_size
            y_end = y_st + tile_size
            if x_st < 0 or y_st < 0:
                continue
            if x_end > width or y_end > height:
                continue

            box = np.zeros((1, 4))
            box[0, 0] = x_st
            box[0, 1] = y_st
            box[0, 2] = x_end
            box[0, 3] = y_end

            # crop out the tile
            pixels = img[y_st:y_end, x_st:x_end]

            # create a full resolution mask, which will be downsampled later after random cropping
            intersection = compute_intersection(box[0, :], bbStar)
            tmp_bbStar = bbStar[intersection > 0, :]
            new_boxes = np.zeros((tmp_bbStar.shape[0], 5), dtype=np.int32)
            # loop over the tile to determine which pixels belong to the foreground
            for i in range(tmp_bbStar.shape[0]):
                bx_st = int(tmp_bbStar[i, 0] - x_st)  # local pixels coordinate
                by_st = int(tmp_bbStar[i, 1] - y_st)  # local pixels coordinate
                bx_end = int(tmp_bbStar[i, 2] - x_st)  # local pixels coordinate
                by_end = int(tmp_bbStar[i, 3] - y_st)  # local pixels coordinate

                bx_st = max(0, bx_st)
                by_st = max(0, by_st)
                bx_end = min(bx_end, pixels.shape[1])
                by_end = min(by_end, pixels.shape[0])

                new_boxes[i, 0] = bx_st
                new_boxes[i, 1] = by_st
                new_boxes[i, 2] = bx_end - bx_st + 1
                new_boxes[i, 3] = by_end - by_st + 1
                new_boxes[i, 4] = tmp_bbStar[i, 4] # copy over the class id of the box

            # only include blocks with foreground, as this is synthetic data using a finite number of backgrounds, we will see the full scope of background when using only foregrounds
            if new_boxes.shape[0] > 0:
                img_list.append(pixels)
                box_list.append(new_boxes)
    return (img_list, box_list)


def generate_database(csv_files):
    print('Generating train data crops')
    output_image_lmdb_file = os.path.join(output_filepath, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    for slide_idx in range(len(csv_files)):
        txn_nb = 0
        csv_file_name = csv_files[slide_idx]
        block_key = csv_file_name.replace('.csv','')
        print('Slide {}/{}'.format(slide_idx, len(csv_files)))
        slide_name = csv_file_name.replace('.csv', '.tif')

        bbStar = load_saved_positions(os.path.join(csv_filepath, csv_file_name))
        full_slide = read_image(os.path.join(image_filepath, slide_name))

        img_list, box_list = process_slide_tiling(full_slide, bbStar)

        for j in range(len(img_list)):
            img = img_list[j]
            boxes = box_list[j]
            key_str = '{}_{:08d}'.format(block_key, txn_nb)
            txn_nb += 1
            write_img_to_db(image_txn, img, boxes, key_str)  # 0 for background label

        image_txn.commit()
        image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()


def generate_block_list(csv_files):
    with open(os.path.join(output_filepath, database_name, 'block_list.csv'), 'w') as fh:
        wr = csv.writer(fh)
        for block_str in csv_files:
            wr.writerow([block_str, ])


if __name__ == "__main__":
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    # find the image files for which annotations exist
    csv_files = [f for f in os.listdir(csv_filepath) if f.endswith('.csv')]


    # csv_files = csv_files[0:2] # used to build mini-database
    #
    # global database_name
    # database_name = 'database-simulated-yolov3.lmdb'
    # generate_database(csv_files)

    # csv_files = csv_files[0:5]  # used to build mini-database

    idx = int(0.8 * len(csv_files))
    train_csv_files = csv_files[0:idx]
    test_csv_files = csv_files[idx:]

    global database_name
    database_name = 'train-database-yolov3.lmdb'
    generate_database(train_csv_files)
    generate_block_list(train_csv_files)

    database_name = 'test-database-yolov3.lmdb'
    generate_database(test_csv_files)
    generate_block_list(test_csv_files)


