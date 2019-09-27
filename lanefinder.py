import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import os

class LaneFinder:
    def __init__(self, verbose=False):
        print('Initialising LaneFinder')
        # image processing parameters
        self.sobel_ksize = 5
        self.sobelx_th = (6, 12)
        self.sobelg_th = (np.pi/4-np.pi/9, np.pi/4+np.pi/9) # 45 +/- 20 [deg]
        self.canny_th = (32, 128)
        self.sat_th = (64, 255)
        self.lit_th = (175, 255)
        self.bee_th = (150, 200)
        self.ell_th = (200, 255)

        # line fit parameters
        self.nwindows = 10
        self.margin = 100 # window half-width, from current centre
        self.minpix = 500 # minimum nonzero pixel in window for recentring
        self.smooth = 1 # how many frames (maximum) to smooth coefficients over
        self.buffersize = 100 # how many prior coefficients to keep
        self.hiswin = 480 # histogram y-window for sans prior lane fit
        
        # metric conversion
        self.xmppx = 3.7/700
        self.ymppx = 3/158
        
        # Sanity check
        if self.buffersize < self.smooth:
            self.buffersize = self.smooth
        
        # parameters for perspective transform. From bottom right counter-clockwise
        # these are read out manually from a sample undistorted image
        self.pers_src = np.float32([(1093, 719),
                                    ( 724, 475),
                                    ( 559, 475),
                                    ( 208, 719)])

        # pick these for a nice dimension
        self.pers_dst = np.float32([(1000, 720),
                                    (1000, 0),
                                    ( 300, 0),
                                    ( 300, 720)])
        
        # where to put saved images
        self.outdir = 'output_images'
        
        self.calibrate(verbose)
        # get perspective transform matrix and its inverse
        self.permat = cv2.getPerspectiveTransform(self.pers_src, self.pers_dst)
        self.invmat = cv2.getPerspectiveTransform(self.pers_dst, self.pers_src) # to reverse the perspective transform later
        self.reset()
        
        
    def reset(self):
        self.has_prior = False
        self.lcoef_buffer = []
        self.rcoef_buffer = []
        
        self.lcoef_metric_buffer = []
        self.rcoef_metric_buffer = []
        
        
    def calibrate(self, verbose=False):
        calims = [os.path.join('camera_cal', fn) for fn in os.listdir('camera_cal')]
        
        ncol, nrow = (9, 6)
        objpts = np.array([[x, y, 0] for y in range(nrow) for x in range(ncol)], np.float32)

        impts = []
        obpts = []

        for fname in calims:
            im = mpim.imread(fname)
            ret, corners = cv2.findChessboardCorners(im, (ncol, nrow))
            if ret:
                impts.append(corners)
                obpts.append(objpts)

        self.ret, self.mtx, self.dsc, self.tvx, self.rvx =\
        cv2.calibrateCamera(obpts, impts, im.shape[1::-1], None, None)
        print('Calibration complete')
        
        # show one calibration result
        if verbose:
            src = mpim.imread(calims[2])
            dst = cv2.undistort(src, self.mtx, self.dsc, None, self.mtx)
            
            plt.figure(figsize=(8, 4.5));plt.imshow(src);plt.gca().set_title('original')
            plt.axis('off');plt.savefig(os.path.join(self.outdir, '00_distorted.jpg'), bbox_inches='tight')
            plt.figure(figsize=(8, 4.5));plt.imshow(dst);plt.gca().set_title('undistorted')
            plt.axis('off');plt.savefig(os.path.join(self.outdir, '01_undistorted.jpg'), bbox_inches='tight')
        
        
    def prep_sobel(self, imud, verbose=False):
        # prepare sobel binary
        if len(imud.shape) > 2:
            grey = cv2.cvtColor(imud, cv2.COLOR_RGB2GRAY)
        else:
            grey = imud
            
        grey = cv2.GaussianBlur(grey, (5, 5), 0)

        sobelx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255*sobelx/np.max(sobelx))
        sobelx_bin = np.zeros_like(sobelx)
        sobelx_bin[(sobelx > self.sobelx_th[0]) & (sobelx <= self.sobelx_th[1])] = 1

        sobely = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        sobely = np.absolute(sobely)
        sobely = np.uint8(255*sobely/np.max(sobely))
        #sobely_bin = np.zeros_like(sobely)
        #sobely_bin[(sobely > self.sobelx_th[0]) & (sobely <= self.sobelx_th[1])] = 1
        
        sobelm = np.sqrt(sobelx**2 + sobely**2)
        sobelm = np.uint8(255*sobelm/np.max(sobelm))
        #sobelm_bin = np.zeros_like(sobelm)
        #sobelm_bin[(sobelm > self.sobelx_th[0]) & (sobelm <= self.sobelx_th[1])] = 1

        # find gradient
        sobelg = np.arctan2(sobely, sobelx)
        sobelg_bin = np.zeros_like(sobelg, dtype=np.uint8)
        sobelg_bin[(sobelg > self.sobelg_th[0]) & (sobelg <= self.sobelg_th[1])] = 1

        if verbose: 
            plt.figure(figsize=(16, 9))
            plt.subplot(2, 2, 1);plt.imshow(sobelx, cmap='gray');plt.gca().set_title('sobel x');plt.axis('off')
            plt.subplot(2, 2, 2);plt.imshow(sobely, cmap='gray');plt.gca().set_title('sobel y');plt.axis('off')
            plt.subplot(2, 2, 3);plt.imshow(sobelg_bin, cmap='gray');plt.gca().set_title('sobel gradient');plt.axis('off')
            plt.subplot(2, 2, 4);plt.imshow(sobelx_bin, cmap='gray');plt.gca().set_title('sobel binary');plt.axis('off')
            plt.savefig(os.path.join(self.outdir, '04_sobel.jpg'), bbox_inches='tight', pad_inches=0)
        
        return (sobelx_bin, sobelg_bin)
    
    
    def prep_canny(self, imud, verbose=False):
        grey = cv2.cvtColor(imud, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 0)
        canny = cv2.Canny(blur, self.canny_th[0], self.canny_th[1])
        
        if verbose:
            plt.figure(figsize=(8, 4.5))
            plt.imshow(canny, cmap='gray');plt.gca().set_title('canny')
        
        return canny//255
    
    
    def prep_luv(self, imud, verbose=False):
        imluv = cv2.cvtColor(imud, cv2.COLOR_RGB2LUV)
        imell = imluv[:, :, 0]
        
        imell_bin = np.zeros_like(imell)
        imell_bin[(imell > self.ell_th[0]) & (imell <= self.ell_th[1])] = 1
        
        if verbose:
            plt.figure(figsize=(16, 9))
            plt.subplot(2, 2, 1);plt.imshow(imluv[:, :, 0], cmap='gray');plt.gca().set_title('LUV-L');plt.axis('off')
            plt.subplot(2, 2, 2);plt.imshow(imluv[:, :, 1], cmap='gray');plt.gca().set_title('LUV-U');plt.axis('off')
            plt.subplot(2, 2, 3);plt.imshow(imluv[:, :, 2], cmap='gray');plt.gca().set_title('LUV-V');plt.axis('off')
            plt.subplot(2, 2, 4);plt.imshow(imell_bin, cmap='gray');plt.gca().set_title('LUV-bin');plt.axis('off')
            plt.savefig(os.path.join(self.outdir, '04a_luv.jpg'), bbox_inches='tight', pad_inches=0)
            
        return imell_bin
    
    
    def prep_lab(self, imud, verbose=False):
        imlab = cv2.cvtColor(imud, cv2.COLOR_RGB2LAB)
        imbee = imlab[:, :, 2]
        
        imbee_bin = np.zeros_like(imbee)
        imbee_bin[(imbee > self.bee_th[0]) & (imbee <= self.bee_th[1])] = 1
        
        if verbose:
            plt.figure(figsize=(16, 9))
            plt.subplot(2, 2, 1);plt.imshow(imlab[:, :, 0], cmap='gray');plt.gca().set_title('LAB-L');plt.axis('off')
            plt.subplot(2, 2, 2);plt.imshow(imlab[:, :, 1], cmap='gray');plt.gca().set_title('LAB-A');plt.axis('off')
            plt.subplot(2, 2, 3);plt.imshow(imlab[:, :, 2], cmap='gray');plt.gca().set_title('LAB-B');plt.axis('off')
            plt.subplot(2, 2, 4);plt.imshow(imbee_bin, cmap='gray');plt.gca().set_title('LAB-bin');plt.axis('off')
            plt.savefig(os.path.join(self.outdir, '04b_lab.jpg'), bbox_inches='tight', pad_inches=0)
        
        return imbee_bin        
        
    
    def prep_hls(self, imud, verbose=False):
        # raw to hls
        imhls = cv2.cvtColor(imud, cv2.COLOR_RGB2HLS)

        # apply s-channel threshhold
        imsat = imhls[:, :, 2]
        imsat_bin = np.zeros_like(imsat)
        imsat_bin[(imsat > self.sat_th[0]) & (imsat <= self.sat_th[1])] = 1
        
        imlit = imhls[:, :, 1]
        imlit_bin = np.zeros_like(imlit)
        imlit_bin[(imlit > self.lit_th[0]) & (imlit <= self.lit_th[1])] = 1

        # display each channel
        if verbose:
            plt.figure(figsize=(16, 13.5))
            plt.subplot(3, 2, 1);plt.imshow(imhls[:, :, 2], cmap='gray');plt.gca().set_title('sat');plt.axis('off')
            plt.subplot(3, 2, 2);plt.imshow(imsat_bin, cmap='gray');plt.gca().set_title('sat bin');plt.axis('off')
            plt.subplot(3, 2, 3);plt.imshow(imhls[:, :, 1], cmap='gray');plt.gca().set_title('lit');plt.axis('off')
            plt.subplot(3, 2, 4);plt.imshow(imlit_bin, cmap='gray');plt.gca().set_title('lit bin');plt.axis('off')
            plt.subplot(3, 2, 5);plt.imshow(imhls[:, :, 0], cmap='hsv');plt.gca().set_title('hue');plt.axis('off')
            plt.savefig(os.path.join(self.outdir, '05_hls.jpg'), bbox_inches='tight')
            
        return (imsat_bin, imlit_bin)
    
    
    def fit_lanes_sans_prior(self, imbird_bin, verbose=False, imout=None):
        # Find histogram for initial centres
        # Search for initial centres start at the bottom of the image.
        # Occasionally, no line pixels are found in either side, only noises.
        # This will throw off the rest of the box search.
        # We must then check the x-centres of the inital boxes against this sanity check:
        # They must be about a regulation lane width apart: 3.7 [m].
        # Take the first sane centres, if none found, use the whole height of image.
        # This fallback would fail on striped lanes with high curvature,
        # where the stripe is on the upper part of the bird-eye view perspective
        
        good_centres = False
        ymax = imbird_bin.shape[0]
        ymin = np.max((ymax - self.hiswin, 0))
        xmid = imbird_bin.shape[1]//2
        
        while ymin > 0:
            hist = np.sum(imbird_bin[ymin:ymax, :], 0)
            cur_leftx = np.argmax(hist[:xmid])
            cur_rightx = np.argmax(hist[xmid:]) + xmid
            
            dx = self.xmppx*(cur_rightx - cur_leftx)
            rt = np.abs(dx)/3.7
            
            if verbose:
                print(ymin, ymax)
                print('dx: {}, rt: {}'.format(dx/self.xmppx, rt))
                
            if (rt > 0.90) and (rt < 1.10):
                good_centres = True
                break
            
            ymax = np.max((0, ymax - self.hiswin//2))
            ymin = np.max((0, ymin - self.hiswin//2))
            
        if not good_centres:
            ymid = 0
            hist = np.sum(imbird_bin[ymid:, :], 0)

            # calculate initial centres
            cur_leftx = np.argmax(hist[:xmid])
            cur_rightx = np.argmax(hist[xmid:]) + xmid
        
        xmax = imbird_bin.shape[1]
        winheight = imbird_bin.shape[0]//self.nwindows

        # bookkeeping
        nonzero = np.nonzero(imbird_bin)
        nonzerox = nonzero[1]
        nonzeroy = nonzero[0]
        llane_idx = []
        rlane_idx = []
        
        for i in range(self.nwindows):
            # yrange of window under consideration
            winy_top = imbird_bin.shape[0] - (i+1)*winheight
            winy_bot = imbird_bin.shape[0] - i*winheight

            # xrange of window under consideration (left and right)
            lwinx_l = np.max((0, cur_leftx - self.margin))
            lwinx_r = np.min((xmax, cur_leftx + self.margin))

            rwinx_l = np.max((0, cur_rightx - self.margin))
            rwinx_r = np.min((xmax, cur_rightx + self.margin))

            # draw current window
            if not (imout is None) and verbose:
                cv2.rectangle(imout, (lwinx_l, winy_bot), (lwinx_r, winy_top), [0, 255, 0], 2) # left
                cv2.rectangle(imout, (rwinx_l, winy_bot), (rwinx_r, winy_top), [0, 255, 0], 2) # right

            lgoed = ((nonzerox > lwinx_l) & (nonzerox <= lwinx_r) & \
                     (nonzeroy <= winy_bot) & (nonzeroy > winy_top)).nonzero()[0]
            rgoed = ((nonzerox > rwinx_l) & (nonzerox <= rwinx_r) & \
                     (nonzeroy <= winy_bot) & (nonzeroy > winy_top)).nonzero()[0]

            llane_idx.append(lgoed)
            rlane_idx.append(rgoed)
            
            if len(lgoed) >= self.minpix:
                cur_leftx = np.int(np.mean(nonzerox[lgoed]))
            if len(rgoed) >= self.minpix:
                cur_rightx = np.int(np.mean(nonzerox[rgoed]))

        llane_idx = np.concatenate(llane_idx)
        rlane_idx = np.concatenate(rlane_idx)

        lx = nonzerox[llane_idx]
        ly = nonzeroy[llane_idx]
        rx = nonzerox[rlane_idx]
        ry = nonzeroy[rlane_idx]
            
        return (lx, ly, rx, ry)
    
    
    def fit_lanes_with_prior(self, imbin, verbose=False, imout=None):     
        span = np.min((self.smooth, len(self.lcoef_buffer)))
        lcoef = np.mean(self.lcoef_buffer[-1:(-1-span):-1], 0)
        rcoef = np.mean(self.rcoef_buffer[-1:(-1-span):-1], 0)
        
        nonzero = np.nonzero(imbin)
        nonzerox = nonzero[1]
        nonzeroy = nonzero[0]
        
        prior_lxfit = lcoef[0]*nonzeroy**2 + lcoef[1]*nonzeroy + lcoef[2]
        prior_rxfit = rcoef[0]*nonzeroy**2 + rcoef[1]*nonzeroy + rcoef[2]
        
        lix = ((nonzerox > prior_lxfit-self.margin) & (nonzerox <= prior_lxfit + self.margin)).nonzero()[0]
        rix = ((nonzerox > prior_rxfit-self.margin) & (nonzerox <= prior_rxfit + self.margin)).nonzero()[0]
        
        lx = nonzerox[lix]
        ly = nonzeroy[lix]
        rx = nonzerox[rix]
        ry = nonzeroy[rix]
        
        # draw search space overlay
        if not (imout is None) and verbose:
            imolay = np.zeros_like(imout)
            yfit = np.linspace(0, imolay.shape[0]-1, imolay.shape[0])
            lxfit = lcoef[0]*yfit**2 + lcoef[1]*yfit + lcoef[2]
            rxfit = rcoef[0]*yfit**2 + rcoef[1]*yfit + rcoef[2]
            
            lpts_winl = np.transpose(np.vstack((lxfit-self.margin, yfit)))
            lpts_winr = np.flipud(np.transpose(np.vstack((lxfit+self.margin, yfit))))
            lpts = np.vstack((lpts_winl, lpts_winr))
            
            rpts_winl = np.transpose(np.vstack((rxfit-self.margin, yfit)))
            rpts_winr = np.flipud(np.transpose(np.vstack((rxfit+self.margin, yfit))))
            rpts = np.vstack((rpts_winl, rpts_winr))
            
            cv2.fillPoly(imolay, np.int_([lpts]), (0, 255, 0))
            cv2.fillPoly(imolay, np.int_([rpts]), (0, 255, 0))
            cv2.addWeighted(imout, 1, imolay, 0.5, 0, imout)
        
        return (lx, ly, rx, ry)
    
    def fit_curve(self, imbird_bin, lx, ly, rx, ry):
        # fit pixel
        lcoef = np.polyfit(ly, lx, 2)
        rcoef = np.polyfit(ry, rx, 2)
        
        # fit metric
        lcoef_m = np.polyfit(self.ymppx*ly, self.xmppx*lx, 2)
        rcoef_m = np.polyfit(self.ymppx*ry, self.xmppx*rx, 2)
        
        self.lcoef_buffer.append(lcoef)
        self.rcoef_buffer.append(rcoef)
        
        self.lcoef_metric_buffer.append(lcoef_m)
        self.rcoef_metric_buffer.append(rcoef_m)
        
        if len(self.lcoef_buffer) > self.buffersize:
            self.lcoef_buffer = self.lcoef_buffer[-self.buffersize:]
            self.rcoef_buffer = self.rcoef_buffer[-self.buffersize:]
            self.lcoef_metric_buffer = self.lcoef_metric_buffer[-self.buffersize:]
            self.rcoef_metric_buffer = self.rcoef_metric_buffer[-self.buffersize:]

        idx0 = np.max((-1-len(self.lcoef_buffer), -1-self.smooth))
        lcoef = np.mean(self.lcoef_buffer[-1:idx0:-1], 0)
        rcoef = np.mean(self.rcoef_buffer[-1:idx0:-1], 0)
        
        lcoef_m = np.mean(self.lcoef_metric_buffer[-1:idx0:-1], 0)
        rcoef_m = np.mean(self.rcoef_metric_buffer[-1:idx0:-1], 0)
        
        yfit = np.linspace(0, imbird_bin.shape[0]-1, imbird_bin.shape[0])
        lxfit = lcoef[0]*yfit**2 + lcoef[1]*yfit + lcoef[2]
        rxfit = rcoef[0]*yfit**2 + rcoef[1]*yfit + rcoef[2]
        
        # calculate curvature
        yeval = self.ymppx*np.max(yfit)
        lcurve = (1 + (2*lcoef_m[0]*yeval + lcoef_m[1])**2)**(3/2)/np.abs(2*lcoef_m[0])
        rcurve = (1 + (2*rcoef_m[0]*yeval + rcoef_m[1])**2)**(3/2)/np.abs(2*rcoef_m[0])
        
        # calculate lane centre
        # this is used to calculate how off-centre the car is right now
        # car centre is at centre of image, lane centre is x-average at image bottom
        # the difference is the 'uncentre' of the car
        lane_centre = (lxfit[-1] + rxfit[-1])/2
        uncentre = (imbird_bin.shape[1]/2 - lane_centre)*self.xmppx
        
        return (yfit, lxfit, rxfit, lcurve, rcurve, uncentre)
    
    
    def check_sanity(self, lwidth, lcurve, rcurve, yfit):
        ratio_width = np.abs(lwidth - 3.7)/3.7
        
        ratio_curve = np.abs(lcurve - rcurve)/lwidth
        
        lcoef = self.lcoef_buffer[-1]
        rcoef = self.rcoef_buffer[-1]
        
        lgrad = np.mean(2*lcoef[0]*yfit[-1:-11:-1] + lcoef[1]*yfit[-1:-11:-1])
        rgrad = np.mean(2*rcoef[0]*yfit[-1:-11:-1] + rcoef[1]*yfit[-1:-11:-1])
        ratio_grad = np.abs(lgrad - rgrad)/np.max((np.abs(lgrad), np.abs(rgrad)))
        
        sane_a = ratio_width < 0.25
        sane_b = (ratio_curve > 0.5) and (ratio_curve < 10)
        sane_c = ratio_grad < 2
        
        # two or more sanity check pass, I'll accept it
        sane = sane_a and (sane_b or sane_c)
        return (sane, ratio_width, ratio_curve, (lgrad, rgrad, ratio_grad))
        
    
    def find_lane_lines(self, im, ignore_prior=False, verbose=False):
        # undistort image
        imud = cv2.undistort(im, self.mtx, self.dsc, None, self.mtx)
        if verbose:
            plt.figure(figsize=(8, 4.5));plt.imshow(im);plt.gca().set_title('input image')
            plt.axis('off');plt.savefig(os.path.join(self.outdir, '02_input.jpg'), bbox_inches='tight', pad_inches=0)
            plt.figure(figsize=(8, 4.5));plt.imshow(imud);plt.gca().set_title('undistorted')
            plt.axis('off');plt.savefig(os.path.join(self.outdir, '03_undistorted.jpg'), bbox_inches='tight', pad_inches=0)

        # prepare sobel binary
        #(sobelm_bin, sobelg_bin) = self.prep_sobel(imud, verbose)
        
        # prepare canny
        # canny = self.prep_canny(imud, verbose)
        
        # prepare hls binary
        #(imsat_bin, imlit_bin) = self.prep_hls(imud, verbose)
        
        # prepare lab binary
        imlab_bin = self.prep_lab(imud, verbose)
        
        # prepare luv binary
        imluv_bin = self.prep_luv(imud, verbose)
        
        # merge binaries
        #imcmb_bin = (imlab_bin | imluv_bin | sobelm_bin) & sobelg_bin
        imcmb_bin = imlab_bin | imluv_bin

        ymax, xmax = imcmb_bin.shape        
        
        # get bird-eye view image
        imbird_bin = cv2.warpPerspective(imcmb_bin, self.permat, (xmax, ymax), flags=cv2.INTER_LINEAR)

        if verbose:
            imbird = cv2.warpPerspective(imud, self.permat, (xmax, ymax), flags=cv2.INTER_LINEAR)
            plt.figure(figsize=(16, 4.5))
            plt.subplot(1, 2, 1);plt.imshow(imbird);plt.gca().set_title('bird-eye view')
            plt.subplot(1, 2, 2);plt.imshow(imbird_bin, cmap='gray');plt.gca().set_title('bird-eye view binary')
            plt.savefig(os.path.join(self.outdir, '06_birdeye.jpg'), bbox_inches='tight', pad_inches=0)

        imout = np.dstack([imbird_bin, imbird_bin, imbird_bin])*255
        
        # find lane points
        if not self.has_prior or ignore_prior:
            lx, ly, rx, ry = self.fit_lanes_sans_prior(imbird_bin, verbose, imout)
            if (len(lx)) < 1 or (len(ly) < 1) or (len(rx) < 1) or (len(ry) < 1):
                (sobelm_bin, sobelg_bin) = self.prep_sobel(imud, verbose)
                imcmb_bin = (imcmb_bin | sobelm_bin) & sobelg_bin
                imbird_bin = cv2.warpPerspective(imcmb_bin, self.permat, (xmax, ymax), flags=cv2.INTER_LINEAR)
                imout = np.dstack([imbird_bin, imbird_bin, imbird_bin])*255
                lx, ly, rx, ry = self.fit_lanes_sans_prior(imbird_bin, verbose, imout)
        else:
            lx, ly, rx, ry = self.fit_lanes_with_prior(imbird_bin, verbose, imout)
            if (len(lx)) < 1 or (len(ly) < 1) or (len(rx) < 1) or (len(ry) < 1):
                (sobelm_bin, sobelg_bin) = self.prep_sobel(imud, verbose)
                imcmb_bin = (imcmb_bin | sobelm_bin) & sobelg_bin
                imbird_bin = cv2.warpPerspective(imcmb_bin, self.permat, (xmax, ymax), flags=cv2.INTER_LINEAR)
                imout = np.dstack([imbird_bin, imbird_bin, imbird_bin])*255
                lx, ly, rx, ry = self.fit_lanes_sans_prior(imbird_bin, verbose, imout)
        if (len(lx)) < 1 or (len(ly) < 1) or (len(rx) < 1) or (len(ry) < 1):
            cv2.putText(imud, 'FATAL FAILURE', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            return imud
        
        # fit curve
        yfit, lxfit, rxfit, lcurve, rcurve, uncentre = self.fit_curve(imbird_bin, lx, ly, rx, ry)
        
        # gather quantities for sanity check
        lwidth = (rxfit[-1] - lxfit[-1])*self.xmppx
        sane, ratio_width, ratio_curve, xgrad = self.check_sanity(lwidth, lcurve, rcurve, yfit)
        lgrad, rgrad, ratio_grad = xgrad
        
        # if result is not sane, try again ignoring prior
        if not sane:
            lx, ly, rx, ry = self.fit_lanes_sans_prior(imbird_bin, verbose, imout)
            if (len(lx)) < 1 or (len(ly) < 1) or (len(rx) < 1) or (len(ry) < 1):
                cv2.putText(imud, 'FATAL FAILURE', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                return imud
            yfit, lxfit, rxfit, lcurve, rcurve, uncentre = self.fit_curve(imbird_bin, lx, ly, rx, ry)
            lwidth = (rxfit[-1] - lxfit[-1])*self.xmppx
            sane, ratio_width, ratio_curve, xgrad = self.check_sanity(lwidth, lcurve, rcurve, yfit)
            lgrad, rgrad, ratio_grad = xgrad
            # if still not sane, accept it. We'll get a sane one eventually (right?)

        # show result
        if verbose:               
            imout[ly, lx] = [255, 0, 0]
            imout[ry, rx] = [0, 0, 255]
            plt.figure(figsize=(8, 4.5))
            plt.plot(lxfit, yfit, 'c-')
            plt.plot(rxfit, yfit, 'm-')
            plt.imshow(imout)
            prior_status = 'with' if self.has_prior else 'sans'
            plt.gca().set_title('fit output {} prior'.format(prior_status))
            plt.savefig(os.path.join(self.outdir, '07_fit_output_{}_prior.jpg'.format(prior_status)), bbox_inches='tight', pad_inches=0)
            
        if not self.has_prior and sane:
            self.has_prior = True
        else:
            self.has_prior = False

        # draw polygon bound by the fitted lane lines
        lpts = np.transpose(np.vstack((lxfit, yfit)))
        rpts = np.flipud(np.transpose(np.vstack((rxfit, yfit))))
        ppts = np.vstack((lpts, rpts))

        impoly = np.zeros_like(imud)
        cv2.fillPoly(impoly, np.int_([ppts]), (0, 255, 0))

        # inverse perspective to camera view (undistorted)
        impolypers = cv2.warpPerspective(impoly, self.invmat, (imud.shape[1], imud.shape[0]))

        # sum images
        imsum = cv2.addWeighted(imud, 1, impolypers, 0.5, 0)
        
        # add helpful texts
        cv2.putText(imsum, 'curves {: 10,.2f}, {: 10,.2f} [m], delta {: 10,.2f}'.format(lcurve, rcurve, ratio_curve), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imsum, 'uncentre {:+5.2f} [m]'.format(uncentre), 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imsum, 'lanewidth {:+5.2f} [m]'.format(lwidth), 
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imsum, 'lgrad {: 10,.2f}, rgrad {: 10,.2f} ratio {: 10,.2f}'.format(lgrad, rgrad, ratio_grad), 
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if sane:
            cv2.putText(imsum, '**__sane**', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(imsum, '**insane**', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if verbose:
            plt.figure(figsize=(8, 4.5));plt.imshow(imsum)
            plt.axis('off');plt.savefig(os.path.join(self.outdir, '08_final_output.jpg'), bbox_inches='tight', pad_inches=0)

        return imsum
