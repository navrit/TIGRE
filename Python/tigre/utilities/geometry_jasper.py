from __future__ import division
import numpy as np
from tigre.utilities.geometry import Geometry

class ConeGeometryJasper(Geometry):

    def __init__(self, high_quality=True, nVoxel = None):

        Geometry.__init__(self)
        if high_quality:
            # VARIABLE                                          DESCRIPTION                    UNITS
            # -------------------------------------------------------------------------------------
            self.DSD = 192.57                                      # Distance Source Detector      (mm)
            self.DSO = self.DSD-43.74                                       # Distance Source Origin        (mm)
            # Detector parameters
            self.nDetector = np.array((512, 512))               # number of pixels              (px)
            self.dDetector = np.array((0.055, 0.055))               # size of each pixel            (mm)
            self.sDetector = self.nDetector * self.dDetector    # total size of the detector    (mm)
            # Image parameters
            self.nVoxel = np.array((512, 512, 512))             # number of voxels              (vx)
            self.sVoxel = np.array((28.16, 28.16, 28.16))          # total size of the image       (mm)
            self.dVoxel = self.sVoxel/self.nVoxel               # size of each voxel            (mm)
            # Offsets
#             self.offOrigin = np.array((0,2.53,0))                # Offset of image from origin   (mm)
            self.offOrigin = np.array((0,0,0))                # Offset of image from origin   (mm)
            self.offDetector = np.array((0, 0))                 # Offset of Detector            (mm)
#             self.offDetector = np.array((0, -2.4684))                 # Offset of Detector            (mm)
            self.rotDetector = np.array((0,0,0))

            # Auxiliary
            self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
            # Mode
            self.mode = 'cone'                                  # parallel, cone                ...
            self.filter = None
        else:
            self.DSD = 192.57                                      # Distance Source Detector      (mm)
            self.DSO = self.DSD -  43.74                                       # Distance Source Origin        (mm)
            # Detector parameters
#             self.nDetector = np.array((256, 256))               # number of pixels              (px)
            self.nDetector = np.array((11, 512))               # number of pixels              (px)
            self.dDetector = np.array((0.055, 0.055))               # size of each pixel            (mm)
            self.sDetector = self.nDetector * self.dDetector    # total size of the detector    (mm)
            # Image parameters
            self.nVoxel = np.array((1, 512, 512))             # number of voxels              (vx)
            self.sVoxel = np.array((0.05, 25.6, 25.6))          # total size of the image       (mm)
            self.dVoxel = self.sVoxel/self.nVoxel               # size of each voxel            (mm)
            # Offsets
#             self.offOrigin = np.array((0,2.53,0))                # Offset of image from origin   (mm)
            self.offOrigin = np.array((0,0,0))                # Offset of image from origin   (mm)
            self.offDetector = np.array((0,0))                 # Offset of Detector            (mm)
            self.rotDetector = np.array((0,0,0))

            # Auxiliary
            self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
            # Mode
            self.mode = 'cone'                                  # parallel, cone                ...
            self.filter = None
        if nVoxel is not None:
            # VARIABLE                                          DESCRIPTION                    UNITS
            # -------------------------------------------------------------------------------------
            self.DSD = 1536                                     # Distance Source Detector      (mm)
            self.DSO = 1000                                     # Distance Source Origin        (mm)
                                                                # Detector parameters
            self.nDetector = np.array((nVoxel[1],
                                       nVoxel[2])
                                                                ) # (V,U) number of pixels        (px)
            self.dDetector = np.array([0.8, 0.8])               # size of each pixel            (mm)
            self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)
                                                                # Image parameters
            self.nVoxel = np.array((nVoxel))                    # number of voxels              (vx)
            self.sVoxel = np.array((256, 256, 256))             # total size of the image       (mm)
            self.dVoxel = self.sVoxel / self.nVoxel             # size of each voxel            (mm)
            # Offsets
            self.offOrigin = np.array((0, 0, 0))                # Offset of image from origin   (mm)
            self.offDetector = np.array((0, 0))                 # Offset of Detector            (mm)
            self.rotDetector = np.array((0, 0, 0))
            # Auxiliary
            self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
            # Mode
            self.mode = 'cone'                                  # parallel, cone
