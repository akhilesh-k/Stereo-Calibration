import numpy as np
import os,cv2,yaml,argparse
from glob import glob

def splitfn(x):
    path,name = os.path.split(x)
    name,ext = os.path.splitext(x)
    return path,name,ext

def coords(s,t,tt):
    try:
        x, y = map(tt, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("%s must be x,y" % t)

def recaberror(img_points,obj_points,rvec,tvec,camera_matrix,dist_coeffs):
    #Compute mean of reprojection error
    reprojected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected_points = reprojected_points.reshape(-1,2)
    tot_error =np.sum(np.abs(img_points-reprojected_points)**2)
    total_points =len(obj_points)

    mean_error=np.sqrt(tot_error/total_points)
    return mean_error

def loadcalib(name):
    o = yaml.load(open(name,"rb"))
    o["camera_matrix"] = np.array(o["camera_matrix"],dtype=np.float32)
    o["dist"]= np.array(o["dist"],dtype=np.float32) 
    return o
def main():
    parser = argparse.ArgumentParser(description='Camera Calibrator - OpenCV and Emanuele Ruffaldi SSSA 2014-2015')
    parser.add_argument('path', help='path where images can be found (png or jpg)',nargs="+")
    parser.add_argument('--save', help='name of output calibration in YAML otherwise prints on console')
    parser.add_argument('--verbose',action="store_true")
    parser.add_argument('--ir',action='store_true')
    parser.add_argument('--threshold',type=int,default=0)
    parser.add_argument('--calib',type=str,help="default calib for guess or nocalibrate mode")
    parser.add_argument('--flipy',action='store_true')
    parser.add_argument('--flipx',action='store_true')
    parser.add_argument('--side',help="side: all,left,right",default="all")
    #parser.add_argument('--load',help="read intrinsics from file")
    #parser.add_argument('--nocalibrate',action="store_true",help="performs only reprojection")
    #parser.add_argument('--noextract',action="store_true",help="assumes features already computed (using yaml files and not the images)")
    parser.add_argument('--debug',help="debug dir for chessboard markers",default="")
    parser.add_argument('--pattern_size',default=(6,9),help="pattern as (w,h)",type=lambda s: coords(s,'Pattern',int))
    parser.add_argument('--square_size2',default=(0,0),help="alt square size",type=lambda s: coords(s,'Square Size',float))
    parser.add_argument('--grid_offset',default=(0,0),help="grid offset",type=lambda s: coords(s,'Square Size',float))
    parser.add_argument('--target_size',default=None,help="target image as (w,h) pixels",type=lambda s: coords(s,'Target Image',int), nargs=2)
    parser.add_argument('--aperture',default=None,help="sensor size in m as (w,h)",type=lambda s: coords(s,'Aperture',float), nargs=2)
    parser.add_argument('--square_size',help='square size in m',type=float,default=0.025)
    parser.add_argument('--nodistortion',action="store_true");
    parser.add_argument('--outputpath',help="path for output yaml files");
    parser.add_argument('--nocalibrate',action="store_true")
    args = parser.parse_args()

    # From documentation http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # C default of cvFindChessboardCorners is ADAPT+NORM
    # Python default of cvFindChessboardCorners is ADAPT 
    if args.ir:
        eflags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    else:
        eflags = cv2.CALIB_CB_ADAPTIVE_THRESH
    #eflags += cv2.CALIB_CB_FAST_CHECK #+ cv2.CV_CALIB_CB_FILTER_QUADS
    #CV_CALIB_CB_FILTER_QUADS
    if False:
        if args.intrinsics != None:
            # load yaml
            pass
        if args.nocalibrate:
            pass
        if args.noextract:
            pass

    if args.calib:
        calib = loadcalib(args.calib)
    else:
        calib = dict(camera_matrix=None,dist=None)


    img_names = []
    for p in args.path:
        img_names.extend(glob(p))
    debug_dir = args.debug
    square_size = args.square_size
    print ("square_size is",square_size)

    pattern_size_cols_rows = (args.pattern_size[0],args.pattern_size[1])
    pattern_points = np.zeros( (np.prod(pattern_size_cols_rows), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size_cols_rows).T.reshape(-1, 2)

    if args.flipy:
        for i in range(0,pattern_points.shape[0]):
            pattern_points[i,1] = args.pattern_size[1] - pattern_points[i,1] - 1

    if args.flipx:
        for i in range(0,pattern_points.shape[0]):
            pattern_points[i,0] = args.pattern_size[0] - pattern_points[i,0] - 1

    # Non square patterns, broadcast product making a non-square grid
    if args.square_size2[0] != 0:
        pattern_points *= np.array([args.square_size2[0],args.square_size2[1],0.0])
    else:
        pattern_points *= args.square_size
    if args.grid_offset[0] != 0 or args.grid_offset[1] != 0:
        pattern_points[:,0] += args.grid_offset[0]
        pattern_points[:,1] += args.grid_offset[1]


    obj_points = []
    img_points = []
    yaml_done = []
    target = args.target_size
    h, w = 0, 0
    lastsize = None
    criteriasub = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteriacal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.001)

    #giacomo
    #for both sub and cal cv::TermCriteria term_criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, DBL_EPSILON);

    print ("images",img_names)
    j = 0
    img_namesr = []
    for fn in (sorted(img_names)):
        if os.path.isdir(fn):
            for y in os.listdir(fn):
                if y.endswith(".jpg") or y.endswith(".png"):
                    img_namesr.append(os.path.join(fn,y))
        else:
            img_namesr.append(fn)
    img_names = img_namesr

    for fn in (sorted(img_names)):
        if fn.endswith(".yaml"):
            continue
        j = j +1
        print (fn,'processing',j)
        img = cv2.imread(fn,-1)
        if img is None:
          print (fn,"failed to load")
          continue
        h, w = img.shape[:2]
        if args.side == "left":
            if len(img.shape)== 3:
                img = img[0:h,0:w/2,:]
            else:
                img = img[0:h,0:w/2]
            w = w/2
        elif args.side == "right":
            if len(img.shape)== 3:
                img = img[0:h,w/2:,:]
            else:
                img = img[0:h,w/2:]
            w = w/2

        if target is not None and (h,w) != target:
            print (fn, (h,w),"->",target)
            img = cv2.resize(img,target)
            h,w = target
        else:
            if lastsize is None:
                lastsize = (h,w)
                print ("using",(h,w))
            else:
                if lastsize != (h,w):
                    print (fn, "all images should be the same size, enforcing")
                    target = lastsize
                    img = cv2.resize(img,target)
                    h,w = target
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            pattern_size_cols_rows
        if args.threshold > 0:
            retval,img = cv2.threshold(img, args.threshold, 255, cv2.THRESH_BINARY);
            print ("thresholded ",img.shape,gray.dtype)
            cv2.imshow("ciao",img)
            cv2.waitKey(0)
        #255-gray if we flipped it
        found, corners = cv2.findChessboardCorners(img, pattern_size_cols_rows,flags=eflags)
        if found:
            # Giacomo (11,11)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteriasub)
        if not found:
            print (fn,'chessboard not found')
            continue
        if args.outputpath:
            yamlfile = os.path.join(args.outputpath,os.path.splitext(os.path.split(fn)[1])[0]+".yaml")
        else:
            yamlfile = os.path.splitext(fn)[0]+".yaml"

        info = dict(width=w,height=h,image_points=corners.reshape(-1,2).tolist(),world_points=pattern_points.tolist())
        yaml.dump(info,open(yamlfile,"wb"))
        print ("\tgenerated yaml")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
        yaml_done.append(yamlfile)
        if debug_dir is not None:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size_cols_rows, corners, found)
            path,name = os.path.split(fn)
            name,ext = os.path.splitext(name)
            dd = '%s/%s_chess.png' % (debug_dir, name)
            cv2.imwrite(dd, vis)
            print ("\twriting debug",dd)

    if not args.nocalibrate:    

        #CV_CALIB_USE_INTRINSIC_GUESS
        if len(obj_points) == 0:
            print ("cannot find corners")
            return
        flags = 0
        if args.nodistortion:
            flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_ZERO_TANGENT_DIST
        if args.calib:
            flags = flags  | cv2.CV_CALIB_USE_INTRINSIC_GUESS
        print ("calibrating...",len(img_points),"images")
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), calib["camera_matrix"], calib["dist"],criteria=criteriacal,flags=flags)
        print ("error:", rms)
        print ("camera matrix:\n", camera_matrix)
        print ("distortion coefficients:", dist_coefs.transpose())
        #cv2.destroyAllWindows()

        #apertureWidth
        #apertureHeight
        if args.aperture:
            fovx,fovy,focalLength,principalPoint,aspectRatio = cv2.calibrationMatrixValues(camera_matrix,(w,h),args.aperture[0],args.aperture[1])



        outname = args.save
        if outname is not None:
            ci = dict(image_width=w,image_height=h,pattern_size=list(pattern_size_cols_rows),rms=rms,camera_matrix=camera_matrix.tolist(),dist=dist_coefs.ravel().tolist(),square_size=square_size)
            print (ci)
            yaml.dump(ci,open(outname,"wb"))

        for i,y in enumerate(yaml_done):
            o = yaml.load(open(y,"rb"))
            o["rms"] = float(recaberror(img_points[i],obj_points[i],rvecs[i],tvecs[i],camera_matrix,dist_coefs))
            o["rvec"] = rvecs[i].tolist()
            o["tvec"] = tvecs[i].tolist()
            yaml.dump(o,open(y,"wb"))
    elif calib["camera_matrix"] is not None:
        for i,y in enumerate(yaml_done):
            retval,rvec,tvec = cv2.solvePnP(obj_points[i], img_points[i], calib["camera_matrix"],calib["dist"])
            o = yaml.load(open(y,"rb"))
            o["rms"] = float(recaberror(img_points[i],obj_points[i],rvec,tvec,calib["camera_matrix"],calib["dist"]))
            o["rvec"] = rvec.tolist()
            o["tvec"] = tvec.tolist()
            yaml.dump(o,open(y,"wb"))


if __name__ == '__main__':
    main()
