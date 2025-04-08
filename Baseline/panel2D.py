#coding=utf-8
# 2D panel code to calculate the pressure distribution and lift force a 2D foil
# Has the functionality to generate 2D foil geometry

# Created by YJ-Wang for students' work
# 2024-06-19 
import numpy
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import math

class panel2D:
    def __init__(self, point1, point2) -> None:
        self._start = numpy.array(point1)
        self._end = numpy.array(point2)
        self.update()
    
    def update(self) -> None:
        self._dir = self._end - self._start
        self._length = (self._dir[0]**2 + self._dir[1]**2)**0.5
        self._dir /= self._length
        self._normal = numpy.array([-self._dir[1], self._dir[0]])
        self._center = 0.5 * (self._start + self._end)   

    def uniformSourceInduced(self, pos):
        pos = numpy.array(pos)
        x = numpy.sum((pos - self._start) * self._dir)
        z = numpy.sum((pos - self._start) * self._normal)
        if abs(z) < 1e-6:
            z = 1e-6
        x2 = self._length # x1 = 0
        xz1 = x**2+z**2
        xz2 = (x-x2)**2+z**2
        tmp = numpy.arctan2(z*x2, (x-x2)*x + z**2)

        phi = (x*numpy.log(xz1) - (x-x2)*numpy.log(xz2) -2*x2 + 2*z*tmp)/(4*numpy.pi)
        u = numpy.log(xz1/xz2) / (4*numpy.pi)
        w = tmp / (2*numpy.pi)
        vel = u * self._dir + w * self._normal
        return phi, vel
    
    def uniformDoubletInduced(self, pos):
        pos = numpy.array(pos)
        x = numpy.sum((pos - self._start) * self._dir)
        z = numpy.sum((pos - self._start) * self._normal)
        if abs(z) < 1e-6:
            z = 1e-6
        x2 = self._length # x1 = 0
        xz1 = x**2+z**2
        xz2 = (x-x2)**2+z**2
        tmp = numpy.arctan2(z*x2, (x-x2)*x + z**2)

        phi = -tmp/(2 * numpy.pi)
        u = -(z/xz1 - z/xz2)/(2 * numpy.pi)
        w = (x/xz1 - (x-x2)/xz2)/(2 * numpy.pi)
        vel = u * self._dir + w * self._normal
        return phi, vel
    
    def uniformSourceInducedFar(self, pos):
        pos = numpy.array(pos)
        r2 = sum((pos - self._center)**2)
        phi = numpy.log(r2**0.5)/(2*numpy.pi)
        vel = (pos - self._center) / r2 / (2*numpy.pi)
        return phi, vel
    
    def uniformDoubletInducedFar(self, pos):
        pos = numpy.array(pos)
        r2 = sum((pos - self._center)**2)
        rVector = pos - self._center
        phi = -rVector[1] / r2 / (2 * numpy.pi)
        u = rVector[0]*rVector[1] / r2**2 / numpy.pi
        w = -(rVector[0]**2 - rVector[1]**2) / r2**2 / (2 * numpy.pi)
        vel = u * self._dir + w * self._normal
        return phi, vel

class foilGeom2D():
    def __init__(self) -> None:
        self._x = None
        self._y = None

    def validate(self):
        if self._x is None or self._y is None:
            raise RuntimeError("Foil not built.")
        if len(self._x) != len(self._y):
            raise RuntimeError("Inconsistent coordinate data.")
        return True

    def build(self, camberRatio, thickRatio, chordLength = 1.0):
        '''
        build the 2D foil geometry with naca 66(Mod) thickness distribution and naca a=0.8(Mod) camber form.

        Parameters
        ==========
        camberRatio: the max camber to chord ratio, f_max/C
        thickRatio: the max thickness to chord ratio, t_max/C
        chordLength: the chord length C
        '''

        xrate = [0, .005, .0075, .0125, .025, .05, .075, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60,
                     .65, .70, .75, .80, .85, .90, .95, 1.0]

        yrate = [.0000, .00281, .00396, .00603, .01055, .01803, .02432, .02981, .03903, .04651, .05257, .05742,
                    .06120, .06394, .06571, .06651, .06631, .06508, .06274, .05913, .05401, .04673, .03607, .02452,
                    .01226, .0000]
        tanf = [0, .47539, .44004, .39531, .33404, .27149, .23378, .20618, .16546, .13452, .10873, .08595,
                .06498, .04507, .02559, .00607, -.01404, -.03537, -.05887, -.08610, -.12058, -.18034, -.23430,
                -.24521, -.24521, -.24521]
        
        trate = [0, .0665, .0812, .1044, .1466, .2066, .2525, .2907, .3521, .4000, .4363, .4637, .4832, .4952,
                    .5000, .4962, .4846, .4653, .4383, .4035, .3612, .3110, .2532, .1877, .1143, 0] # .0333]


        maxf = max(yrate)

        xc = numpy.array(xrate) * chordLength
        yc = numpy.array(yrate)/maxf * (camberRatio * chordLength)
        yt = numpy.array(trate) * thickRatio * chordLength # maxt is 0.5
        realTanf = (numpy.array(tanf)/maxf * camberRatio)
        cosf = 1 / (realTanf **2 + 1) ** 0.5
        sinf = cosf * realTanf
        xu = xc - yt*sinf
        yu = yc + yt*cosf
        xl = xc + yt*sinf
        yl = yc - yt*cosf

        self._x = numpy.array(xl.tolist()[::-1] + xu.tolist()[1:])
        self._y = numpy.array(yl.tolist()[::-1] + yu.tolist()[1:])

        return self._x, self._y
    
    def plot(self):
        self.validate()
        plt.plot(self._x, self._y, '-*')
        plt.axis("equal")
        plt.grid()
        plt.show()

    def rotateDegree(self, angle):
        '''
        Rotate the foil with a degree

        Parameters
        ==========
        angle: the rotating angle in degree
        '''
        angRad = math.radians(angle)
        tx = self._x * math.cos(angRad) + self._y * math.sin(angRad)
        ty = -self._x * math.sin(angRad) + self._y * math.cos(angRad)
        self._x = tx
        self._y = ty

    def reDiscretize(self, halfNumber = 50):
        midIndex = int(len(self._x) / 2)
        xu = self._x[:midIndex + 1]
        yu = self._y[:midIndex + 1]
        xl = self._x[midIndex:]
        yl = self._y[midIndex:]
        su = [0]
        sl = [0]
        for i in range(midIndex):
            su.append(su[-1] + ((xu[i] - xu[i+1])**2 + (yu[i] - yu[i+1])**2)**0.5)
            sl.append(sl[-1] + ((xl[i] - xl[i+1])**2 + (yl[i] - yl[i+1])**2)**0.5)
        su_new = (1 - numpy.cos(numpy.linspace(0, math.pi, halfNumber))) * 0.5 * su[-1]
        sl_new = (1 - numpy.cos(numpy.linspace(0, math.pi, halfNumber))) * 0.5 * sl[-1]
        xu = itp.CubicSpline(su, xu)(su_new)
        yu = itp.CubicSpline(su, yu)(su_new)
        xl = itp.CubicSpline(sl, xl)(sl_new)
        yl = itp.CubicSpline(sl, yl)(sl_new)
        self._x = numpy.array(xu.tolist()[:-1] + xl.tolist())
        self._y = numpy.array(yu.tolist()[:-1] + yl.tolist())

class solver():
    def __init__(self) -> None:
        None

    def solve(self, angleInDegree, camberRatio):
        chord = 1
        thickRatio = 0.1
        vel0 = numpy.array([1.0, 0.0])

        ## 1. build geometry
        geom = foilGeom2D()
        geom.build(camberRatio = camberRatio, thickRatio = thickRatio, chordLength=chord)
        geom.rotateDegree(angleInDegree)
        geom.reDiscretize()
        # geom.plot()

        ## 2. build panels
        points = [[x, y] for x, y in zip(geom._x, geom._y)] + [[200., geom._y[-1]]]
        panels = []
        for i in range(len(points)-1):
            panels.append(panel2D(points[i], points[i+1]))

        ## 3. build matrix
        numPanel = len(panels)-1
        Amat = numpy.zeros((numPanel, numPanel))
        Bmat = numpy.zeros((numPanel, numPanel))
        sigma = numpy.zeros((numPanel,))
        Cmat = numpy.zeros((numPanel,))
        for i in range(numPanel):
            for j in range(numPanel):
                phiS, velS = panels[j].uniformSourceInduced(panels[i]._center)
                Bmat[i,j] = phiS
                if i == j:
                    Amat[i, j] = 0.5
                    continue
                phiD, velD = panels[j].uniformDoubletInduced(panels[i]._center)
                Amat[i,j] = phiD
            phiD, velD = panels[-1].uniformDoubletInduced(panels[i]._center)
            Cmat[i] = phiD
            sigma[i] = -numpy.sum(panels[i]._normal * vel0)
        Amat[:, -1] += Cmat
        Amat[:, 0] -= Cmat
        
        rhs = [-numpy.sum(Bmat[i, :] * sigma) for i in range(numPanel)]
        mu = numpy.linalg.solve(Amat, rhs)

        # xArray = [p._center[0] for p in panels[:-1]]
        # plt.plot(xArray, mu)
        # plt.show()

        ### velocity calculation option 1
        velOption = 2
        if velOption == 1:
            muW = mu[-1] - mu[0]
            velArray = []
            for i in range(numPanel):
                vel = numpy.zeros((2,))
                for j in range(numPanel):
                    phiS, velS = panels[j].uniformSourceInduced(panels[i]._center)
                    phiD, velD = panels[j].uniformDoubletInduced(panels[i]._center)
                    vel += velS * sigma[j] + velD * mu[j]
                phiD, velD = panels[-1].uniformDoubletInduced(panels[i]._center)
                vel += muW * velD
                vel += vel0
                velArray.append(vel)
        else:
            ### velocity calculation option 2
            velArray = []
            for i in range(numPanel):
                if i == 0:
                    vel = -(mu[1] - mu[0]) / (0.5*panels[0]._length + 0.5*panels[1]._length) * panels[i]._dir
                elif i == numPanel - 1:
                    vel = -(mu[i] - mu[i-1]) / (0.5*panels[i]._length + 0.5*panels[i-1]._length) * panels[i]._dir
                else:
                    l1 = 0.5*panels[i]._length + 0.5*panels[i-1]._length
                    l2 = 0.5*panels[i]._length + 0.5*panels[i+1]._length
                    derivative = (mu[i+1] * l1**2 - mu[i-1] * l2**2 - mu[i] * (l1**2 - l2**2)) / (l1*l2**2 + l2*l1**2)
                    vel = -derivative * panels[i]._dir
                vel += numpy.sum(vel0 * panels[i]._dir) * panels[i]._dir
                velArray.append(vel)
        
        # for i in range(numPanel):
        #     print(f"{i}-th panel normal vel:", numpy.sum(velArray[i] * panels[i]._normal), numpy.sum(panels[i]._dir * panels[i]._normal))
        
        ######## output
        CpArray = []
        liftForce = 0
        dragForce = 0
        vel02 = numpy.sum(vel0**2)
        for i in range(numPanel):
            l = panels[i]._length
            n = panels[i]._normal
            vel = velArray[i]
            pressure = 0.5 * (vel02 - numpy.sum(vel**2))
            Cp = pressure / (0.5 * vel02)
            CpArray.append([panels[i]._center[0], Cp])
            liftForce += -pressure * l * n[1]
            dragForce += -pressure * l * n[0]
            # print(f"{l:>.6f} ({n[0]:>9.6f}, {n[1]:>9.6f}) ({vel[0]:>9.6f}, {vel[1]:>9.6f}) {Cp:>9.6f} {force[0]:>9.6f} {force[1]:>9.6f}")

        self._CpArray = numpy.array(CpArray)
        self._Cl = liftForce / (0.5 * vel02 * chord) # lift coefficient
        self._Cd = dragForce / (0.5 * vel02 * chord) # drag coefficient

        return self._CpArray, self._Cl, self._Cd
    
    def plotCp(self):
        plt.plot(self._CpArray[:, 0], self._CpArray[:, 1])
        plt.xlabel('x')
        plt.ylabel('Cp')
        plt.show()

    def plotCp_save(self,savepth = './cp.jpg'):
        plt.plot(self._CpArray[:, 0], self._CpArray[:, 1])
        plt.xlabel('x')
        plt.ylabel('Cp')
        plt.savefig(savepth)
        plt.show()
    
        


class test:
    @staticmethod
    def testInducedFunction():
        start = [-0.5, 0]
        end = [0.5, 0]
        p = panel2D(start, end)
        yArray = numpy.array([-10, -5, -2, -1, -0.5, -0.2, -0.1, -0.01, 0.01, 0.2, 0.5, 1, 2, 5, 10])
        phiSArray = []
        phiS1Array = []
        for y in yArray:
            pos = numpy.array([0.5*y, y])
            print(f"== position: [{pos[0]}, {pos[1]}]")
            phiS, velS = p.uniformSourceInduced(pos)
            phiSArray.append(phiS)
            phiS1, velS1 = p.uniformSourceInducedFar(pos)
            phiS1Array.append(phiS1)
            print("  source:", phiS, velS, phiS1, velS1)
            phiD, velD = p.uniformDoubletInduced(pos)
            phiD1, velD1 = p.uniformDoubletInducedFar(pos)
            print("  doublet:", phiD, velD, phiD1, velD1)
        print(numpy.array(phiSArray) - numpy.array(phiS1Array))

    @staticmethod
    def testGeometry(angleInDegree, camberRatio):
        chord = 1
        thickRatio = 0.1
        geom = foilGeom2D()
        geom.build(camberRatio = camberRatio, thickRatio = thickRatio, chordLength=chord)
        geom.rotateDegree(angleInDegree)
        geom.reDiscretize()
        geom.plot()


if __name__ == "__main__":
    s = solver()
    s.solve(angleInDegree=1.2247980833053589, camberRatio=0.03999999910593033)
    print(f"lift coefficient: {s._Cl:>.3f}, drag coefficient: {s._Cd:>.3f}")
    # print(s._CpArray)
    #print(s._CpArray[numpy.argmax(s._CpArray[:,1]),0])
    s.plotCp() 
    ## standard 1.9 0.06
    # 1.2247980833053589, camber ratio: 0.03999999910593033, Using Steps: 17 cl: 0.705
    # 1.726742148399353, camber ratio: 0.09322938323020935, Using Steps: 7  cl: 1.501
