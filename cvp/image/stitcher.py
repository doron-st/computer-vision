# import the necessary packages
import cv2.cv2 as cv2
import numpy as np


class Stitcher:

    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = True

    def stitch(self, images, ratio=0.75, reproj_thresh=4.0, show_matches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (image_b, image_a) = images
        (kps_a, features_a) = self.detect_and_describe(image_a)
        (kps_b, features_b) = self.detect_and_describe(image_b)
        # match features between the two images
        M = self.match_keypoints(kps_a, kps_b,
                                 features_a, features_b, ratio, reproj_thresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M
        result = cv2.warpPerspective(image_a, H,
                                     (image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
        result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

        # check to see if the keypoint matches should be visualized
        if show_matches:
            vis = self.draw_matches(image_a, image_b, kps_a, kps_b, matches,
                                    status)
            # return a tuple of the stitched image and the visualization
            return result, vis
        # return the stitched image
        return result

    def detect_and_describe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X

        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPyarrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return kps, features

    def match_keypoints(self, kps_a, kps_b, features_a, features_b, ratio, reproj_thresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, 2)
        matches = []
        # loop over the raw matches
        for m in raw_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            pts_a = np.float32([kps_a[i] for (_, i) in matches])
            pts_b = np.float32([kps_b[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reproj_thresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return matches, H, status
        # otherwise, no homograpy could be computed
        return None

    def draw_matches(self, image_a, image_b, kps_a, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = image_a.shape[:2]
        (hB, wB) = image_b.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image_a
        vis[0:hB, wA:] = image_b
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                pt_a = (int(kps_a[queryIdx][0]), int(kps_a[queryIdx][1]))
                pt_b = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, pt_a, pt_b, (0, 255, 0), 1)
        # return the visualization
        return vis
