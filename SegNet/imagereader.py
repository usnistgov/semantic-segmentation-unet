
import multiprocessing
from multiprocessing import Process
import queue
import random
import traceback
import lmdb
import numpy as np
import augment
import os
import skimage.io
import skimage.transform
from isg_ai_pb2 import ImageYoloBoxesPair
import yolov3


def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    std = np.std(image_data)
    mv = np.mean(image_data)
    if std <= 1.0:
        # normalize (but dont divide by zero)
        image_data = (image_data - mv)
    else:
        # z-score normalize
        image_data = (image_data - mv) / std

    return image_data


def imread(fp):
    return skimage.io.imread(fp, as_gray=True)


class ImageReader:

    def __init__(self, img_db, batch_size, use_augmentation=True, balance_classes=False, shuffle=True, num_workers=1):
        self.image_db = img_db
        self.use_augmentation = use_augmentation
        self.queue_starvation = False
        if balance_classes:
            raise Exception('Class balancing currently not implemented')

        if not os.path.exists(self.image_db):
            print('Could not load database file: ')
            print(self.image_db)
            raise Exception("Missing Database")

        self.shuffle = shuffle

        random.seed()

        # get a list of keys from the lmdb
        self.keys_flat = list()

        self.lmdb_env = lmdb.open(self.image_db, map_size=int(2e10), readonly=True) # 20 GB
        self.lmdb_txns = list()

        datum = None
        print('Initializing image database')
        with self.lmdb_env.begin(write=False) as lmdb_txn:
            cursor = lmdb_txn.cursor()
            for key, _ in cursor:
                self.keys_flat.append(key)

                if datum is None:
                    datum = ImageYoloBoxesPair()  # create a datum for decoding serialized caffe_pb2 objects
                    # extract the serialized image from the database
                    value = lmdb_txn.get(key)
                    # convert from serialized representation
                    datum.ParseFromString(value)

        self.image_size = [datum.img_height, datum.img_width]

        self.batchsize = batch_size
        self.nb_workers = num_workers
        # setup queue
        self.terminateQ = multiprocessing.Queue(maxsize=self.nb_workers)  # limit output queue size
        self.maxOutQSize = 100
        self.outQ = multiprocessing.Queue(maxsize=self.maxOutQSize)  # limit output queue size
        self.idQ = multiprocessing.Queue(maxsize=self.nb_workers)

        self.workers = None
        self.done = False

    def get_epoch_size(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat) / self.batchsize)

    def get_image_size(self):
        return self.image_size

    def startup(self):
        self.workers = None
        self.done = False

        [self.idQ.put(i) for i in range(self.nb_workers)]
        [self.lmdb_txns.append(self.lmdb_env.begin(write=False)) for i in range(self.nb_workers)]
        # launch workers
        self.workers = [Process(target=self.__image_loader) for i in range(self.nb_workers)]

        # start workers
        for w in self.workers:
            w.start()

    def shutdown(self):
        # tell workers to shutdown
        for w in self.workers:
            self.terminateQ.put(None)

        # empty the output queue (to allow blocking workers to terminate
        nb_none_received = 0
        # empty output queue
        while nb_none_received < len(self.workers):
            try:
                while True:
                    val = self.outQ.get_nowait()
                    if val is None:
                        nb_none_received += 1
            except queue.Empty:
                pass  # do nothing

        # wait for the workers to terminate
        for w in self.workers:
            w.join()


    @staticmethod
    def __format_image(image_data):
        # reshape into tensor (NCHW)
        image_data = image_data.reshape((-1, 1, image_data.shape[0], image_data.shape[1]))
        return image_data

    @staticmethod
    def __format_boxes(boxes, img_size):
        # reshape into tensor [grid_size, grid_size, num_anchors, 5 + num_classes]

        anchors = np.asarray(yolov3.ANCHORS, dtype=np.float32)
        num_anchors = len(anchors)
        num_classes = yolov3.NUMBER_CLASSES



        grid_sizes = []
        grid_sizes.append(int(img_size / yolov3.NETWORK_DOWNSAMPLE_FACTOR))
        grid_sizes.append(int(img_size / (yolov3.NETWORK_DOWNSAMPLE_FACTOR / 2)))
        grid_sizes.append(int(img_size / (yolov3.NETWORK_DOWNSAMPLE_FACTOR / 4)))
        num_layers = len(grid_sizes)

        label = []
        for l in range(num_layers):
            # leading 1 dimension is the batch, later these will be concatentated together along that dimension
            label.append(np.zeros((1, grid_sizes[l], grid_sizes[l], num_anchors, (5 + num_classes)), dtype=np.float32))

        if boxes is None:
            return label

        boxes = boxes.astype(np.float32)

        box_xy = boxes[:, 0:2]
        box_wh = boxes[:, 2:4]

        # move box x,y to middle from upper left
        box_xy = np.floor(box_xy + ((box_wh-1) / 2.0))
        boxes[:, 0:2] = box_xy
        boxes[:, 2:4] = box_wh

        anchors_max = anchors / 2.0
        anchors_min = -anchors_max
        # set the center of all boxes as the origin of their coordinates
        # and correct their coordinates
        box_wh = np.expand_dims(box_wh, -2)
        boxes_max = box_wh / 2.0
        boxes_min = -boxes_max

        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = box_wh[..., 0] * box_wh[..., 1]

        anchor_area = anchors[:, 0] * anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                i = np.floor(boxes[t, 0] / img_size * grid_sizes[l]).astype('int32')
                j = np.floor(boxes[t, 1] / img_size * grid_sizes[l]).astype('int32')

                c = boxes[t, 4].astype('int32')

                # first dimension is the batch
                label[l][0, j, i, n, 0:4] = boxes[t, 0:4]
                label[l][0, j, i, n, 4] = 1.0
                label[l][0, j, i, n, 5 + c] = 1.0

        return label

    def __get_next_key(self):
        if self.shuffle:
            # if self.balance_classes:
            #     # select a class to add at random from the set of classes
            #     label_idx = random.randint(0, self.number_labels - 1)  # randint has inclusive endpoints
            #     # randomly select an example from the database of the required label
            #     nb_examples = len(self.keys[label_idx])
            #     img_idx = random.randint(0, nb_examples - 1)
            #     # lookup the database key for loading the image data
            #     fn = self.keys[label_idx][img_idx]
            # else:

            # select a key at random from the list (does not account for class imbalance)
            fn = self.keys_flat[random.randint(0, len(self.keys_flat) - 1)]
        else:  # no shuffle
            # without shuffle you cannot balance classes
            fn = self.keys_flat[self.key_idx]
            self.key_idx += self.nb_workers
            self.key_idx = self.key_idx % len(self.keys_flat)

        return fn

    def __image_loader(self):
        termimation_flag = False  # flag to control the worker shutdown
        self.key_idx = self.idQ.get()  # setup non-shuffle index to stride across flat keys properly
        try:
            datum = ImageYoloBoxesPair()  # create a datum for decoding serialized caffe_pb2 objects

            local_lmdb_txn = self.lmdb_txns[self.key_idx]

            # while the worker has not been told to terminate, loop infinitely
            while not termimation_flag:

                # poll termination queue for shutdown command
                try:
                    if self.terminateQ.get_nowait() is None:
                        termimation_flag = True
                        break
                except queue.Empty:
                    pass  # do nothing

                # allocate tensors for the current batch
                pixel_tensor = list()
                label_tensor_1 = list()
                label_tensor_2 = list()
                label_tensor_3 = list()

                # build a batch selecting the labels using round robin through the shuffled order
                for i in range(self.batchsize):
                    fn = self.__get_next_key()

                    # extract the serialized image from the database
                    value = local_lmdb_txn.get(fn)
                    # convert from serialized representation
                    datum.ParseFromString(value)

                    # convert from string to numpy array
                    I = np.fromstring(datum.image, dtype=np.uint8)
                    # reshape the numpy array using the dimensions recorded in the datum
                    I = I.reshape(datum.img_height, datum.img_width)
                    assert (datum.img_height == datum.img_width), 'Images must be same width and height'

                    Boxes = None
                    # construct mask from list of boxes
                    if datum.box_count > 0:
                        # convert from string to numpy array
                        Boxes = np.fromstring(datum.boxes, dtype=np.int32)
                        # reshape the numpy array using the dimensions recorded in the datum
                        Boxes = Boxes.reshape(datum.box_count, 5)
                        # boxes are [x, y, width, height, class-id]

                    if self.use_augmentation:
                        # setup the image data augmentation parameters
                        reflection_flag = True
                        jitter_augmentation_severity = 0.1  # x% of a FOV
                        noise_augmentation_severity = 0.02  # vary noise by x% of the dynamic range present in the image
                        scale_augmentation_severity = 0.1 # vary size by x%
                        blur_max_sigma = 2 # pixels
                        # intensity_augmentation_severity = 0.05

                        I = I.astype(np.float32)

                        # perform image data augmentation
                        I, Boxes = augment.augment_image_box_pair(I, Boxes,
                            reflection_flag=reflection_flag,
                              jitter_augmentation_severity=jitter_augmentation_severity,
                              noise_augmentation_severity=noise_augmentation_severity,
                              scale_augmentation_severity=scale_augmentation_severity,
                              blur_augmentation_max_sigma=blur_max_sigma)

                    # format the image into a tensor
                    I = self.__format_image(I)
                    I = zscore_normalize(I)

                    # convert the boxes into the format expected by yolov3
                    label = self.__format_boxes(Boxes, datum.img_height)
                    label_1, label_2, label_3 = label

                    # append the image and label to the batch being built
                    pixel_tensor.append(I)
                    label_tensor_1.append(label_1)
                    label_tensor_2.append(label_2)
                    label_tensor_3.append(label_3)

                # convert the list of images into a numpy array tensor ready for tensorflow
                pixel_tensor = np.concatenate(pixel_tensor, axis=0).astype(np.float32)
                label_tensor_1 = np.concatenate(label_tensor_1, axis=0).astype(np.float32)
                label_tensor_2 = np.concatenate(label_tensor_2, axis=0).astype(np.float32)
                label_tensor_3 = np.concatenate(label_tensor_3, axis=0).astype(np.float32)

                # add the batch in the output queue
                # this put block until there is space in the output queue (size 50)
                self.outQ.put((pixel_tensor, label_tensor_1, label_tensor_2, label_tensor_3))

        except Exception as e:
            print('***************** Reader Error *****************')
            print(e)
            traceback.print_exc()
            print('***************** Reader Error *****************')
        finally:
            # when the worker terminates add a none to the output so the parent gets a shutdown confirmation from each worker
            self.outQ.put(None)

    def get_batch(self):
        # get a ready to train batch from the output queue and pass to to the caller
        if self.outQ.qsize() < int(0.1*self.maxOutQSize):
            if not self.queue_starvation:
                print('Input Queue Starvation !!!!')
            self.queue_starvation = True
        if self.queue_starvation and self.outQ.qsize() > int(0.5*self.maxOutQSize):
            print('Input Queue Starvation Over')
            self.queue_starvation = False
        return self.outQ.get()

    def generator(self):
        while True:
            batch = self.get_batch()
            if batch is None:
                return
            yield batch

    def get_queue_size(self):
        return self.outQ.qsize()

