#script scons pour compiler sous unix
import os
import socket
import distutils.util

platform = distutils.util.get_platform()
hostname = socket.gethostname() 
print platform
print hostname
src = [Glob('*.cpp')]
env = Environment(CPPPATH='./')
if platform[:6] == 'macosx':
 	print "Nous sommes sur un mac!"
 	env.Replace(LIBS  = '',
 	            CPPFLAGS= '')

if platform[:5] == 'linux':
 	print "Nous sommes sur linux!"
 	env.Replace(CPPFLAGS='-I/usr/local/cuda/include/',LIBS  = ['OpenCL',''])

env.Program('go',src,CXXPATH='.',FRAMEWORKS='opencl')


#import os
#import socket
#import distutils.util

# env = Environment( 	LIBS=['siloh5', 'z', 'hdf5'],
# 			FORTRANFLAGS = '-cpp -D_LINUX',
# 			F90FLAGS = '-cpp',
# 	            	LINK = 'g++',
# 	            	LIBPATH= '/usr/local/lib',
# 	           	F90PATH = ['/usr/local/include'])

# platform = distutils.util.get_platform()
# hostname = socket.gethostname() 

# OPT = '-O3'

# print platform
# if platform[:5] == 'linux':
# 	print "Nous sommes sur linux!"
# 	env.Append(LIBS = ['gfortran', 'gfortranbegin', 'lapack', 'dfftpack'])

# if platform[:6] == 'macosx':
# 	print "Nous sommes sur un mac!"
# 	SetOption('num_jobs', 2)
# 	env.Append(FORTRANFLAGS = '-D_MAC')
# 	env.Replace(FRAMEWORKS  = 'veclib',
# 	            F90FLAGS = '-D_MAC',)
# 	env.Append(LIBS = ['gfortran', 'lapack', 'dfftpack'])

# if hostname[:8] == 'irma-hpc':
# 	print "Nous sommes sur irma-hpc!"
# 	env.Replace(	F90PATH = ['/usr/local/include'])
# 	env.Replace(	LIBPATH= '/usr/local/lib') 
# 	env.Replace(	LINKFLAGS='-Wl,-rpath=/usr/local/lib')

# if platform[:7] == 'solaris':
# 	#path = ['/opt/oracle/solstudio12.2/bin']
# 	#env.Append (ENV = {'PATH' : path})
# 	env.Replace(	F90PATH = ['/usr/local/hdf5/include'])
# 	env.Replace(	LIBPATH= '/usr/local/hdf5/lib') 
# 	env.Replace(	FORTRANFLAGS = '-fast -fpp',
# 			F90 = 'sunf95',
# 			LINK = 'sunf95',
# 			LIBPATH= '/usr/local/hdf5/lib', 
# 		    	LINKFLAGS='-fast -xlic_lib=sunperf -R/usr/local/hdf5/lib',
# 			F90FLAGS = '-fast -fpp',
# 	            	F90PATH = ['/usr/local/hdf5/include'])
# 	env.Append(LIBS = ['nsl', 'socket'])

# env.Program('picsou', Glob('*.f90')+Glob('*.f'))
# env.Decider('MD5')

