#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import time
from multiprocessing import Value, Lock
from multiprocessing.pool import ThreadPool
from tqdm import tqdm # used to show a progress bar. Example: for x in tqdm(XXX). It records how many loops have been finished (with a bar).
import numpy as np
import requests
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

'''
Warning: If you want to read comments please do not add/delete lines because there are line number guidance in comments
Also please us Pycharm.
'''

### Command Line Parameters ###
def main():
    parser = argparse.ArgumentParser(description='ImageNet image scraper')
    parser.add_argument('-scrape_only_flickr', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-number_of_classes', default=20, type=int) # How many class you want.
    parser.add_argument('-images_per_class', default=750, type=int) # The minimum number of images per class.
    parser.add_argument('-data_root', default='./images', type=str) # where to store the images.
    parser.add_argument('-use_class_list', default=True, type=lambda x: (str(x).lower() == 'true')) # what class can you pick.
    parser.add_argument('-class_list', # Actually class code. For example: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00006484
                        default=['n00006484', 'n00007846', 'n00017222', 'n00021265', 'n03902125',
                                 'n00451635', 'n03082979', 'n01605630', 'n01741943', 'n01877134',
                                 'n01887787', 'n01910747', 'n02131653', 'n02437136', 'n02676938',
                                 'n02694662', 'n02686568', 'n04192698', 'n02773037', 'n04226826'], nargs='*')
    parser.add_argument('-debug', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-multiprocessing_workers', default=50, type=int) # Do the things in parallel.
    args, args_other = parser.parse_known_args()

    if args.debug:
        logging.basicConfig(filename='imagenet_scarper.log', level=logging.DEBUG)

    if len(args.data_root) == 0:
        logging.error("-data_root is required to run downloader!")
        exit()

    # Note:
    # 'X' means X is a variable defined in the program
    # [$ XXXX] means what you input in terminal
    # "X" means X is a name or address
    # Function[XXXXX] means the function name is XXXXX
    # ction*[XXXXX] means XXXXis a package


    # Start:
    # First check what is f-string formatting -> #101 (this is line number. Please jump to see)
    # Reference: https://www.datacamp.com/community/tutorials/f-string-formatting-in-python
    # Example: f'xx_{shit}_xx' shit = 'fat_fuck' is equivalent to xx_fat_fuck_xx, remember to remove '{}'.

    # How the data flows:
    # if input [$ use_class_list true] in command line or doesn't input anything(default true) -> #111
    # then read what your input to parser 'class_list' (should look like n00006484), assign it to 'item' -> #112
    # Then check if 'item' in json file (it store all the class codes in image net). You can't input class which doesn't exist -> #114

    # if you set [$ use_class_list false]. Then read the code of image classes from json file. -> #120
    # The jason file is a dictionary which looks like {"n00004475": {"img_url_count": 8, "flickr_img_url_count": 6, "class_name": "organism"}
    # n00004475 is one of the 'key' of the dic. It is the numeric code for class "organism".
    # read the value of the 'key' (its number of images belong to that class). if big enough then put 'key' into a set 'potential_class_pool' -> #123, 126
    # randomly pick codes from 'potential_class_pool', store them to 'classes_to_scrape'. The number of codes you pick is -> #135
    # defined by 'args.number_of_classes', (default 20). -> #23
    # 'picked_classes_idxes' is used to store random index. -> #133

    # till now the codes are stored in 'classes_to_scrape' -> #135
    # And you print it out -> #137
    # jump jump jump many lines -> #341
    # read codes from 'classes_to_scrape' and assign them to 'class_wnid' -> #341
    # tqdm is bar function it shows how many codes(class) has been processed. -> #10
    # send 'class_wnid' to function Function(imagenet_api_wnid_to_urls) -> #344
    # inside the function it inserts the string 'class_wnid' (codes) to a URL -> #102
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={'class_wnid'} -> #102
    # remember it is f-string formatting so remove the {}
    # it actually looks like:
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00004475  (You can click it to see)
    # yeah it has "text" in the url path so the website above has no pictures but the url of the pictures.
    # Now the paths(their corresponding websites contain the paths of all images) are store in 'url_urls' -> #345
    # send request to these website by Function*(requests('url_urls')) and get all the text documents (many url) -> #348
    # store these text in 'resp' -> #348
    # decode the url text and store in 'urls' -> #356
    # look the code here

    #         with ThreadPool(processes=args.multiprocessing_workers) as p: -> #358
    #             p.map(get_image, urls) -> #359

    # parallel processing: send parameter 'urls' to function[get_image] and process them simultaneously -> #359

    # Inside the function, parameter 'urls' is assigned to 'img_url' -> #232
    # use Function*[requests('img_url')] again to send requests to the website -> #289
    # The last time we get url. This time is real images.
    # store images in 'img_resp' -> #289
    # open the folder and write  'img_resp' into it. -> #330
    # All the others are locking in parallel processing and logging system.


    def imagenet_api_wnid_to_urls(wnid):
        return f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}'


    current_folder = os.path.dirname(os.path.realpath(__file__))
    class_info_json_filename = 'imagenet_class_info.json'
    class_info_json_filepath = os.path.join(current_folder, class_info_json_filename)
    with open(class_info_json_filepath) as class_info_json_f:
        class_info_dict = json.load(class_info_json_f)

    classes_to_scrape = []
    if args.use_class_list:
        for item in args.class_list:
            classes_to_scrape.append(item)
            if item not in class_info_dict:
                logging.error(f'Class {item} not found in ImageNete')
                exit()

    elif not args.use_class_list:
        potential_class_pool = []
        for key, val in class_info_dict.items():
            if args.scrape_only_flickr:
                if int(val['flickr_img_url_count']) * 0.9 > args.images_per_class:
                    potential_class_pool.append(key)
            else:
                if int(val['img_url_count']) * 0.8 > args.images_per_class:
                    potential_class_pool.append(key)

        if len(potential_class_pool) < args.number_of_classes:
            logging.error(
                f"With {args.images_per_class} images per class there are {len(potential_class_pool)} to choose from.")
            logging.error(f"Decrease number of classes or decrease images per class.")
            exit()
        picked_classes_idxes = np.random.choice(len(potential_class_pool), args.number_of_classes, replace=False)
        for idx in picked_classes_idxes:
            classes_to_scrape.append(potential_class_pool[idx])

    print("Picked the following clases: \nCount: %s" % len(classes_to_scrape))
    print([class_info_dict[class_wnid]['class_name'] for class_wnid in classes_to_scrape])
    if not os.path.isdir(args.data_root):
        os.mkdir(args.data_root)

    def add_debug_csv_row(row):
        with open('stats.csv', "a") as csv_f:
            csv_writer = csv.writer(csv_f, delimiter=",")
            csv_writer.writerow(row)

    class MultiStats:
        def __init__(self):
            self.lock = Lock()

            self.stats = dict(
                all=dict(
                    tried=Value('d', 0),
                    success=Value('d', 0),
                    time_spent=Value('d', 0),
                ),
                is_flickr=dict(
                    tried=Value('d', 0),
                    success=Value('d', 0),
                    time_spent=Value('d', 0),
                ),
                not_flickr=dict(
                    tried=Value('d', 0),
                    success=Value('d', 0),
                    time_spent=Value('d', 0),
                )
            )

        def inc(self, cls, stat, val):
            with self.lock:
                self.stats[cls][stat].value += val

        def get(self, cls, stat):
            with self.lock:
                ret = self.stats[cls][stat].value
            return ret
    multi_stats = MultiStats()
    if args.debug:
        row = [
            "all_tried",
            "all_success",
            "all_time_spent",
            "is_flickr_tried",
            "is_flickr_success",
            "is_flickr_time_spent",
            "not_flickr_tried",
            "not_flickr_success",
            "not_flickr_time_spent"
        ]
        add_debug_csv_row(row)

    def add_stats_to_debug_csv():
        row = [
            multi_stats.get('all', 'tried'),
            multi_stats.get('all', 'success'),
            multi_stats.get('all', 'time_spent'),
            multi_stats.get('is_flickr', 'tried'),
            multi_stats.get('is_flickr', 'success'),
            multi_stats.get('is_flickr', 'time_spent'),
            multi_stats.get('not_flickr', 'tried'),
            multi_stats.get('not_flickr', 'success'),
            multi_stats.get('not_flickr', 'time_spent'),
        ]
        add_debug_csv_row(row)
    def print_stats(cls, print_func):
        actual_all_time_spent = time.time() - scraping_t_start.value
        processes_all_time_spent = multi_stats.get('all', 'time_spent')

        if processes_all_time_spent == 0:
            actual_processes_ratio = 1.0
        else:
            actual_processes_ratio = actual_all_time_spent / processes_all_time_spent

        print_func(f'STATS For class {cls}:')
        print_func(f' tried {multi_stats.get(cls, "tried")} urls with'
                   f' {multi_stats.get(cls, "success")} successes')

        if multi_stats.get(cls, "tried") > 0:
            print_func(
                f'{100.0 * multi_stats.get(cls, "success") / multi_stats.get(cls, "tried")}% '
                f'success rate for {cls} urls ')
        if multi_stats.get(cls, "success") > 0:
            print_func(
                f'{multi_stats.get(cls, "time_spent") * actual_processes_ratio / multi_stats.get(cls, "success")}'
                f' seconds spent per {cls} successful image download')
    lock = Lock()
    url_tries = Value('d', 0)
    scraping_t_start = Value('d', time.time())
    class_folder = ''
    class_images = Value('d', 0)
    def get_image(img_url):
        def check():
            with lock:
                cls_imgs = class_images.value

            if cls_imgs >= args.images_per_class:
                return True

        if len(img_url) <= 1:
            return

        if check():
            return

        logging.debug(img_url)

        cls = ''

        if 'flickr' in img_url:
            cls = 'is_flickr'
        else:
            cls = 'not_flickr'
            if args.scrape_only_flickr:
                return

        t_start = time.time()

        def finish(status):
            t_spent = time.time() - t_start
            multi_stats.inc(cls, 'time_spent', t_spent)
            multi_stats.inc('all', 'time_spent', t_spent)

            multi_stats.inc(cls, 'tried', 1)
            multi_stats.inc('all', 'tried', 1)

            if status == 'success':
                multi_stats.inc(cls, 'success', 1)
                multi_stats.inc('all', 'success', 1)

            elif status == 'failure':
                pass
            else:
                logging.error(f'No such status {status}!!')
                exit()
            return

        with lock:
            url_tries.value += 1
            if url_tries.value % 250 == 0:
                print(f'\nScraping stats:')
                print_stats('is_flickr', print)
                print_stats('not_flickr', print)
                print_stats('all', print)
                if args.debug:
                    add_stats_to_debug_csv()

        try:
            img_resp = requests.get(img_url, timeout=1)
        except ConnectionError:
            logging.debug(f"Connection Error for url {img_url}")
            return finish('failure')
        except ReadTimeout:
            logging.debug(f"Read Timeout for url {img_url}")
            return finish('failure')
        except TooManyRedirects:
            logging.debug(f"Too many redirects {img_url}")
            return finish('failure')
        except MissingSchema:
            return finish('failure')
        except InvalidURL:
            return finish('failure')

        if 'content-type' not in img_resp.headers:
            return finish('failure')

        if 'image' not in img_resp.headers['content-type']:
            logging.debug("Not an image")
            return finish('failure')

        if len(img_resp.content) < 1000:
            return finish('failure')

        logging.debug(img_resp.headers['content-type'])
        logging.debug(f'image size {len(img_resp.content)}')

        img_name = img_url.split('/')[-1]
        img_name = img_name.split("?")[0]

        if len(img_name) <= 1:
            return finish('failure')

        img_file_path = os.path.join(class_folder, img_name)
        logging.debug(f'Saving image in {img_file_path}')

        if check():
            return

        with open(img_file_path, 'wb') as img_f:
            img_f.write(img_resp.content)

            with lock:
                class_images.value += 1

            logging.debug(f'Scraping stats')
            print_stats('is_flickr', logging.debug)
            print_stats('not_flickr', logging.debug)
            print_stats('all', logging.debug)

            return finish('success')
    print(f"Multiprocessing workers: {args.multiprocessing_workers}")
    for class_wnid in tqdm(classes_to_scrape):

        class_name = class_info_dict[class_wnid]["class_name"]
        url_urls = imagenet_api_wnid_to_urls(class_wnid)

        time.sleep(0.05)
        resp = requests.get(url_urls)

        class_folder = os.path.join(args.data_root, class_name)
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)

        class_images.value = 0

        urls = [url.decode('utf-8') for url in resp.content.splitlines()]

        with ThreadPool(processes=args.multiprocessing_workers) as p:
            p.map(get_image, urls)


if __name__ == '__main__':
    main()
