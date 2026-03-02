import numpy as np


'''
this code is to take the positions and concentration of a particle and due to beer lamberts law and inverse square reverse
the impact they have had on the brightness so we get the true brightnes and all particles are judged fairly,
this way differences in brightness will refer to density of particles rather than just how close they are to the laser
'''
#assumed on a grid where 0,0 is in the centre of the top axis so x=0 is the center axis
def optical_reverse(x, y, brightness):
    #laser sheet intesity, off centeral axis
    def laser_sheet_intensity(x, x0=0, sigma=1):
        ls = np.exp(-(x-x0)**2/(2*sigma**2))
        return ls
        
    def attenuation(x, y, alpha):
        alpha = 5 #attenuation coefficient, this is a constant that depends on the properties of the medium and the particles
        distance = np.sqrt(x**2 + y**2)
        return np.exp(-alpha * distance)
    
    #calculate the true brightness

    true_brightness = brightness / (laser_sheet_intensity(x) * attenuation(x, y, alpha=5))
    return true_brightness


print(optical_reverse(5, -10, 1))


    