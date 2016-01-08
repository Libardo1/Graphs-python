from aqualab.util import put, traverse
import numpy
import matplotlib.pyplot as plt
import sys, os, math
from datetime import datetime, timedelta
from dateutil import parser
import sys,os, shlex, subprocess, json
from collections import OrderedDict
from copy import deepcopy
from aqualab.plot.mCdf import keyedCdf
from matplotlib import cm
from matplotlib.colors import LogNorm
from pialab import utils as myutils
from matplotlib.table import Table
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update({'lines.linewidth':3 })

def setAxLinesBW(ax):
	"""
	Take each Line2D in the axes, ax, and convert the line style to be
	suitable for black and white viewing.
	"""
	MARKERSIZE = 3

	COLORMAP = {
		'b': {'marker': None, 'dash': (None,None)},
		'g': {'marker': None, 'dash': [5,5]},
		'r': {'marker': None, 'dash': [5,3,1,1]},
		'c': {'marker': None, 'dash': [1,3]},
		'm': {'marker': None, 'dash': [5,2,5,2,5,4]},
		'y': {'marker': None, 'dash': [5,3,1,2,1,2]},
		'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
		}
	for line in ax.get_lines() + ax.get_legend().get_lines():
		origColor = line.get_color()
		line.set_color('black')
		line.set_dashes(COLORMAP[origColor]['dash'])
		line.set_marker(COLORMAP[origColor]['marker'])
		line.set_markersize(MARKERSIZE)


def plotcdfo(cdfdata, labels, title, gfolder, xlabel='', figwidth=8, figlen=5, resolution=0.0001, logscalex=False):
	cdf = keyedCdf(baseName=gfolder+'/'+title, resolution=resolution,figsize=(figwidth,figlen),xlabel =  xlabel)
	for v in range(len(labels)):
		label = labels[v]
		data = cdfdata[v]
		label = label +'(' + str(len(data))+')'
		for d in data:
			cdf.insert(label, float(d))
	if logscalex:
		cdf.plot(gfolder,"logscalex",numSymbols=0)
	else:
		cdf.plot(gfolder)


def plotvlines(cdfdata, labels, filename,gfolder, xlabel='',ylabel='', islog=False,figwidth=6, figlen=4, resolution=0.0001, neglect=[], showlen=False,colors=True, legendposition = 'lower right', title =None,xlimit=None,topdata=None, mindatalen = 1,extension='.pdf',xrange=None):
	return
	fig = plt.figure(figsize=(figwidth,figlen))
	plt.axvline(x=5, ymax=10)
	plt.axvline(x=10, ymax=100)
	plt.axvline(x=0, ymax=5)
	plt.show()
	plt.ylim([0,100])
	empty = True
	markervalues = ['*','o','.']#,'_', '^','x','s','v', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'D', 'd', '_' ]
	colorvalues = ['b','g', 'r', 'k', 'y', 'm', 'c','#CCCCCC', '#808080']
	dashes = [[]]#[3,3,3,3], [],[3, 3, 3, 3], [], [3, 3, 3, 3], [5, 1, 5, 1]]
	leng = 0
	maxx = 0
	try:
		if topdata != None:
			mindatalen = min(sorted([len(x) for x in cdfdata], reverse=True)[:topdata])
	except:
		mindatalen = 1
	i = 0
	numdata = len(labels)
	for v in range(len(labels)):
		vindex = v
		color = colorvalues[v%len(colorvalues)]
		if topdata != None:
			color = colorvalues[i%len(colorvalues)]
		marker = markervalues[v%len(markervalues)]
		dash = dashes[v%len(dashes)]
		label = labels[v]
		dat = cdfdata[v]
		i+=1
		#try:
		for d in dat:
			print dat.index(d)*numdata+vindex,d
			plt.axvline(x=d, ymax=float(d/max(dat)))
			continue
			plt.axvline(x=dat.index(d)*numdata+vindex,ymin=0,ymax=d, color=color)
			#, label=label,color=color,dashes=dash)#, marker=marker)#, dashes=dash)#+'('+str(len(data))+')'
		#except:
		#pass
	#plt.legend(loc=legendposition, fontsize=10)
	if xlimit != None:
	 	plt.xlim(xlimit)
	if xrange != 0:
		plt.xlim(xrange)
	def setFigLinesBW(fig):
		"""
		Take each axes in the figure, and for each line in the axes, make the
		line viewable in black and white.
		"""
		for ax in fig.get_axes():
			setAxLinesBW(ax)
	if not colors:
		setFigLinesBW(fig)
	plt.locator_params(nbins=5, axis='x')
	if title != None:
		plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	title = filename.replace(' ','')
	if islog:
		plt.gca().set_xscale('log')
	plt.tight_layout()
	plt.savefig(gfolder+'/'+title+extension)
	plt._show()
	plt.close()

def plotcdf(cdfdata, labels, filename,gfolder, xlabel='', islog=False,figwidth=4, figlen=3, resolution=0.0001, neglect=[], showlen=False,colors=True, legendposition = 'lower right', title =None,xlimit=None,topdata=None, mindatalen = 1,extension='.pdf',xrange=None):
	fig = plt.figure(figsize=(figwidth,figlen))
	empty = True
	markervalues = ['*','o','.']#,'_', '^','x','s','v', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'D', 'd', '_' ]
	colorvalues = ['b','c', 'r','g', 'k', 'y', 'm', 'c','#CCCCCC', '#808080']
	dashes = [[],[3,5,3,5], [],[3, 5, 3, 5], [], [3, 5, 3, 5], [3, 5, 3, 5]]
	leng = 0
	maxx = 0
	try:
		if topdata != None:
			mindatalen = min(sorted([len(x) for x in cdfdata], reverse=True)[:topdata])
	except:
		mindatalen = 1
	i = 0
	for v in range(len(labels)):
		color = colorvalues[v%len(colorvalues)]
		if topdata != None:
			color = colorvalues[i%len(colorvalues)]
		marker = markervalues[v%len(markervalues)]
		dash = dashes[v%len(dashes)]
		label = labels[v]
		temdat = [x for x in cdfdata[v] if x not in neglect]
		dat = numpy.sort(temdat)
		if len(dat) < mindatalen:
			continue
		i+=1
		#if sum(dat)/len(dat) * 4 < max(dat):
		#if len(dat) > 50:
		#	dat = dat[:int(0.90*len(dat))]
		#if len(dat) < 0.01 * maxdatasize and len(dat) < 25:
		#	continue
		try:
			data = [min(dat)]
			if maxx < max(dat): maxx = max(dat)
			data.extend(dat)
			empty = False
			leng +=1
			cdfy = (numpy.arange(len(data))/float(len(data)-1))[0:]
			if showlen:
				label += '('+str(len(data))+')'
			plt.plot(data, cdfy, label=label,color=color,dashes=dash)#, marker=marker)#, dashes=dash)#+'('+str(len(data))+')'
		except:
			pass
	if empty or (leng == 1 and len(cdfdata) != 1):
		plt.close()
		return
	plt.legend(loc=legendposition, fontsize=11)
	if xlimit != None:
	 	plt.xlim(xlimit)
	if xrange != 0:
		plt.xlim(xrange)
	def setFigLinesBW(fig):
		"""
		Take each axes in the figure, and for each line in the axes, make the
		line viewable in black and white.
		"""
		for ax in fig.get_axes():
			setAxLinesBW(ax)
	if not colors:
		setFigLinesBW(fig)
	plt.locator_params(nbins=5, axis='x')
	if title != None:
		plt.title(title)
	plt.ylabel('CDF')
	plt.xlabel(xlabel)
	title = filename.replace(' ','')
	if islog:
		plt.gca().set_xscale('log')
	plt.tight_layout()
	plt.savefig(gfolder+'/'+title+extension)
	plt.close()

def plotcdf2D(cdfdata, upperlabel,lowerlabel, filename,gfolder, xlabel='', islog=False,figwidth=4, figlen=3, resolution=0.0001, neglect=[], showlen=False,colors=True, legendposition = 'lower right', title =None,xlimit=None,topdata=None, mindatalen = 1, extension='.pdf', xrange=None):
	fig = plt.figure(figsize=(figwidth,figlen))
	empty = True
	markervalues = ['*','o','.']#,'_', '^','x','s','v', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'D', 'd', '_' ]
	colorvalues = ['b', 'c', 'r','g' ,'k', 'y', 'm', 'c','#CCCCCC', '#808080']
	dashes = [[2, 1, 2, 1], [5, 2, 5, 2], [3, 3, 3, 3], [5, 1, 5, 1]]
	#linestyles = [ ':', '-','--', ':', '_.', '_']
	#maxdatasize = max([len(x) for x in cdfdata])
	leng = 0
	maxx = 0
	try:
		if topdata != None:
			mindatalen = min(sorted([len(x) for x in cdfdata[0]], reverse=True)[:topdata])
	except:
		pass
	i = 0
	for uv in range(len(upperlabel)):
		ul = upperlabel[uv]
		color = colorvalues[i%len(colorvalues)]
		#if topdata != None:
		#	color = colorvalues[i%len(colorvalues)]
		for ll in range(len(lowerlabel)):
			#marker = markervalues[v%len(markervalues)]
			#linestyle = linestyles[ll%len(linestyles)]
			dash = dashes[ll%len(dashes)]
			if len(lowerlabel) - ll == 1:
				dash = []
			llabel = lowerlabel[ll]
			label = ul + '( '+llabel+' )'
			temdat = [x for x in cdfdata[uv][ll] if x not in neglect]
			dat = numpy.sort(temdat)
			if len(dat) < mindatalen:
				continue
			#if sum(dat)/len(dat) * 4 < max(dat):
			#if len(dat) > 50:
			#	dat = dat[:int(0.90*len(dat))]
			#if len(dat) < 0.01 * maxdatasize and len(dat) < 25:
			#	continue
			#try:
			data = [min(dat)]
			if maxx < max(dat): maxx = max(dat)
			data.extend(dat)
			empty = False
			leng +=1
			cdfy = (numpy.arange(len(data))/float(len(data)-1))[0:]
			if showlen:
				label += '('+str(len(data))+')'
			plt.plot(data, cdfy, label=label,color=color,dashes=dash) #linestyle=linestyle)#, marker=marker, dashes=dash)#+'('+str(len(data))+')'
			#except:
			pass
		i+=1
	if empty or (leng == 1 and len(cdfdata) != 1):
		plt.close()
		return
	plt.legend(loc=legendposition, fontsize=11)
	if xlimit != 0:
	 	plt.xlim(xlimit)
	if xrange != 0:
		plt.xlim(xrange)
	def setFigLinesBW(fig):
		"""
		Take each axes in the figure, and for each line in the axes, make the
		line viewable in black and white.
		"""
		for ax in fig.get_axes():
			setAxLinesBW(ax)
	if not colors:
		setFigLinesBW(fig)
	plt.locator_params(nbins=5, axis='x')
	if title != None:
		plt.title(title)
	plt.ylabel('CDF')
	plt.xlabel(xlabel)
	title = filename.replace(' ','').replace(',','')
	if islog:
		plt.gca().set_xscale('log')
	plt.tight_layout()
	#print 'saving to....'+gfolder+'/'+title+extension
	plt.savefig(gfolder+'/'+title+extension)
	plt.close()


def plotcdfn(cdfdata, labels, title, gfolder, xlabel='', islog=False,figwidth=8, figlen=5, resolution=0.0001):
	fig = plt.figure(figsize=(figwidth,figlen))
	
	for v in range(len(labels)):
		label = labels[v]
		dat = numpy.sort(cdfdata[v])
		data = [min(dat)]
		data.extend(dat)
		cdfy = (numpy.arange(len(data))/float(len(data)-1))[0:]
		plt.plot(data, cdfy, label=label)#+'('+str(len(data))+')')
	
	plt.legend(loc='lower right')
	plt.title(title)
	plt.xlabel(xlabel)
	if islog:
		plt.gca().set_xscale('log')
	plt.savefig(gfolder+'/'+title+'.pdf')
	plt.close()

def plotccdf(cdfdata, labels, title, gfolder, xlabel='', islog=False,figwidth=8, figlen=5, resolution=0.0001):
	fig = plt.figure(figsize=(figwidth,figlen))
	
	for v in range(len(labels)):
	
		label = labels[v]
		dat = numpy.sort(cdfdata[v])
		data = [min(dat)]
		data.extend(dat)
		cdfy = (numpy.arange(len(data))/float(len(data)-1))[0:]
		#data = numpy.sort(cdfdata[v])
		#cdfy = (numpy.arange(len(data)+1)/float(len(data)))[1:]
		ccdfy = 1 - cdfy
		#print len(data), len(ccdfy)
		plt.plot(data, ccdfy, label=label)#+'('+str(len(data))+')')
	plt.legend(loc='upper right')
	#plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('CCDF')
	if islog:
		plt.gca().set_xscale('log')
	plt.savefig(gfolder+'/'+title+'.pdf')
	plt.close()

def drawtable(data, rows, columns, title, folder='Table', rowname='',fmt='{:.2f}'):
	nrows = len(rows)
	ncols = len(columns)
	fig, ax = plt.subplots(figsize=(ncols+2,nrows+2))
	ax.set_axis_off()
	rowlc='#F0F0F0'
	collc='#F0F0F0'
	cellc='#FFFFFF'
	ecol='#0000FF'
	font1 = FontProperties()
	font1.set_size('9')
	fontl = font1.copy()
	fontl.set_weight('bold')
	tb = Table(ax)#, bbox=[0.10,0.10,0.90,0.90])
	tb.auto_set_font_size(False)
	#tb.set_fontsize(100.0)
	width, height = 0.95/(ncols+1), 0.95/(nrows+1)
	for i in range(nrows):
		tb.add_cell(i,-1, width*2, height, text=rows[i][:20], loc='right',edgecolor=ecol, facecolor=rowlc,fontproperties=fontl)
	# Column Labels
	for j in range(ncols):
		tb.add_cell(-1, j, width, height/1.5, text=columns[j][:10], loc='center', edgecolor=ecol, facecolor=collc,fontproperties=fontl)
	tb.add_cell(-1,-1, width*2, height/1.5, text=rowname[:10], loc='right',edgecolor=ecol, facecolor=rowlc,fontproperties=fontl)

	# Add cells
	for i in range(len(data)):
		for j in range(len(data[i])):
			val = data[i][j]
			tb.add_cell(i,j,width, height, text=fmt.format(val), loc='center', edgecolor=ecol, facecolor=cellc, fontproperties=font1)
	# Row Labels
	ax.add_table(tb)
	plt.savefig(folder+'/'+title+'.pdf')
	#plt.show()
	#sys.exit(0)
	plt.close()

# Need to correct
def map2D(title, xlabel, ylabel, xv, yv, folder, xlog = False, ylog = False, same = False):

	if len(xv) is 0 or len(yv) is 0:
		return
	#print "\n ", title
	#print "xv: ", xv
	#print "yv: ", yv
	fig = plt.figure(figsize = (12,8))
	plt.scatter(xv, yv,"ro")

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if xlog:
		plt.xscale('log')
	if ylog:
		plt.yscale('log')
	#print min(xv), max(xv)
	#plt.axis([min(xv)/10, max(xv)*10, 0, (float(max(yv)))])
	fn = title
	xm = max(xv)
	ym = max(yv)
	if xm>ym:
		mx = xm
	else:
		mx = ym
	fn = fn.replace(" ", "")
	fn = fn.replace("/","")
	gd = os.path.join(folder, "Map2D")
	createDir(gd)

	fn = os.path.join(gd, fn+".pdf")

	if same == True and xm is not None and ym is not None:
		#print same, xlabel, ylabel, mx
		plt.axis([0,float(mx)+1, 0, float(mx)+1])
	plt.savefig(fn)

	plt.close()

# Need to correct
def linegraph(title, xlabel, ylabel, legends, xv, yvs, folder, xlog = False, ylog = False, same= False,legendposition='upper right'):
	fig = plt.figure(figsize=(10,6))
	gr = False
	for i in range(len(yvs)):
		yv = yvs[i]
		k = legends[i]
		#print xv, yv
		if len(yv) < len(xv):
			for i in range(len(xv) - len(yv)):
				yv.append(0)
		elif len(yv) < len(xv) or len(xv) == 0 or len(yv) == 0:
			continue
		#print xv, yv, len(xv), len(yv)
		try:
			#xv, yv = sortlists(xv, yv)
			plt.plot(xv, yv, label = k)
			gr = True
		except:
			pass
	if not gr:
		try:
			plt.close()
			return
		except:
			return

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc=legendposition, fontsize=10)
	plt.title(title)
	if xlog:
		plt.xscale('log')
	if ylog:
		plt.yscale('log')
	ax = plt.subplot(111)
	#plt.yscale('log')
	fn = title + "Linegraph"
	fn = fn.replace(" ", "")
	fn = fn.replace("/","")
	gd = os.path.join(folder, "Linegraphs")
	myutils.createDir(gd)
	fn = os.path.join(gd, fn+".pdf")
	plt.savefig(fn)
	plt.close()

# Need to correct
def timelinegraph(title, xlabel, ylabel, dms , folder, xlog = False, ylog = False, same= False):

	for gap in range(2, 4):
			#for rem in range(gap):
			fig = plt.figure(figsize=(16,10))
			keys = dms.keys()
			gr = False
			for i in range(len(keys)):
				if i % gap != 0:
					continue
				k = keys[i]
				xv, yv = dms[k]
				#print k
				#print xv
				#print yv
				if len(yv) == 0:
					continue
				try:
					xv, yv = sortlists(xv, yv)
					plt.plot(xv, yv, label=k)
					gr = True
				except:
					pass

			if gr == False:
				try:
					plt.close()
					return
				except:
					return
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			ax = plt.subplot(111)
			ax.legend()
			#plt.yscale('log')
			fn = title + "TimeLinegraph"
			fn = fn.replace(" ", "")
			fn = fn.replace("/","")
			gd = os.path.join(folder, "TimeLinegraphs")
			createDir(gd)


			fn = os.path.join(gd, fn+str(gap)+".pdf")

			plt.savefig(fn)

			plt.close()

# Need to correct
def timelinegraph0(title, xlabel, ylabel,ums,  dms , folder, xlog = False, ylog = False, same= False):
	fig = plt.figure(figsize=(8,5))
	yv, xv = ums
	#print "\ntimelinegraph0"
	#print title
	#print ums
	#print dms
	if len(yv) == 0:
		return
	try:
		#xv, yv = sortlists(xv, yv)
		plt.plot(xv, yv, label="Upload Speed")
	except:
		pass
	yv, xv = dms
	if len(yv) == 0:
		return
	try:
		#xv, yv = sortlists(xv, yv)
		plt.plot(xv, yv, label="Download Speed")
	except:
		pass
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	ax = plt.subplot(111)
	ax.legend()
	#plt.yscale('log')
	fn = title + "TimeLinegraph"
	fn = fn.replace(" ", "")
	fn = fn.replace("/","")
	gd = os.path.join(folder, "TimeLinegraphs")
	createDir(gd)
	fn = os.path.join(gd, fn+".pdf")
	plt.yticks(numpy.arange(0, 35, 5))
	plt.savefig(fn)
	plt.close()

def singlebargraph(xv, yvs,title, folder='Bargraphs',xlabel='', ylabel='',xticks = None):
	rectlist = []
	N = len(xv)
	ind = numpy.arange(N)
	#colors = ['w','r', 'y', 'b', 'g','c']
	if len(yvs) == 0:
		return
	width = 8.0/36
	if xticks == None:
		xticks = xv
	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2, 1.05*height, "", ha='center', va='bottom')
	fig, ax = plt.subplots(figsize = (8, 6))
	rect1 = ax.bar(ind, yvs, width, color = 'b')
	rectlist.append(rect1)
	autolabel(rect1)
	ax.legend()
	#ax.set_title(title)
	ax.set_xticklabels(xv)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.subplots_adjust(bottom=0.15)
	xticks_pos = [0.5*patch.get_width() + patch.get_xy()[0] for patch in rect1]
	plt.xticks(xticks_pos, xticks, ha = "center", rotation='horizontal')
	myutils.createDir(folder)
	plt.savefig(folder+"/" + title+".pdf")
	plt.close()

def bargraphs(xv, yvss,title, folder='Bargraphs',xlabel='', ylabel='', legend=None, rotation = '0',islog=False, stds=None, loc='upper right', ylimit=None,legendposition='upper right',noxlabel=False, thick=False):
	rectlist = []
	N = len(xv)
	ind = numpy.arange(N)
	colors = ['b', 'c', 'r','g' 'm', 'y','k','c']
	if len(yvss[0]) == 0:
		return
	wx = len(xv)
	if wx < 6:
		wx = 6
	width = 6.0/(wx*len(yvss)*2)
	if thick:
		width = width*4
	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2, 1.05*height, "", ha='center', va='bottom')
	fig, ax = plt.subplots(figsize = (6, 4))
	i=0
	for yvs in yvss:
		std = None
		if stds is not None:
			std = stds[i]
			rect1 = ax.bar(ind+width*i, yvs, width, color=colors[i%6], yerr=std, edgecolor="none")
		else:
			rect1 = ax.bar(ind+width*i, yvs, width, color=colors[i%6], edgecolor="none")
		rectlist.append(rect1)
		autolabel(rect1)
		i +=1
	#ax.set_title(title)
	if not noxlabel:
		ax.set_xticklabels(xv)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	if ylimit is not None:
		plt.ylim(ylimit)
	if islog:
		ax.set_yscale('log')
	if legend is not None:
		ax.legend(rectlist, legend, loc=loc)
	plt.subplots_adjust(bottom=0.15)
	if i%2 == 0:
		xticks_pos = [patch.get_xy()[0] for patch in rectlist[i/2]]
	else:
		xticks_pos = [0.5*patch.get_width() + patch.get_xy()[0] for patch in rectlist[i/2]]
	if not noxlabel:
		plt.xticks(xticks_pos, xv, ha = "center", rotation=rotation)
	myutils.createDir(folder)
	filname = title.replace(' ','' )
	plt.savefig(folder+"/" + filname+".pdf")
	plt.close()

def singlestackedbargraph(xv, yvss,title, folder='Bargraphs',xlabel='', ylabel='', legend=None, rotation = '0',islog=False, stds=None, ylimit=None, iscolor=False):
	rectlist = []
	N = len(xv)
	ind = numpy.arange(N)
	colors = ['b', 'g', 'r','k','c','y', '#CC0C0', '0CCCCC']
	#if not iscolor:
	#	colors = ['w', 'w', 'w','w', 'w', 'w', 'w', 'w']
	patterns = ['.']#'"x", "o", "O", ".", "*", "/" , "\\" , "|" , "-" , "+" ,  ]

	if len(yvss[0]) == 0:
		return
	if len(xv) > 5:
		width = 6.0/(len(xv)*2)
	else:
		width = 6.0/10
	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2, 1.05*height, "", ha='center', va='bottom')
	fig, ax = plt.subplots(figsize = (6, 4))
	i=0
	bottom = [0 for x in range(len(yvss[0]))]
	for yvs in yvss:
		std = None
		if stds is not None:
			std = stds[i]
			rect1 = ax.bar(ind, yvs, width, color=colors[i%6], bottom=bottom,yerr=std, hatch=patterns[i%len(patterns)])
		else:
			rect1 = ax.bar(ind, yvs, width, color=colors[i%6], bottom=bottom)#, hatch=patterns[i%len(patterns)]*3)
		rectlist.append(rect1)
		autolabel(rect1)
		i +=1
		newbottom = [bottom[x] + yvs[x] for x in range(len(bottom))]
		bottom = newbottom
	#ax.set_title(title)
	ax.set_xticklabels(xv)
	plt.ylabel(ylabel)
	if ylimit is not None:
		plt.ylim(ax, ylimit)
	plt.xlabel(xlabel)
	if islog:
		ax.set_yscale('log')
	if legend is not None:
		ax.legend(rectlist, legend, loc='upper right')
	plt.subplots_adjust(bottom=0.15)
	xticks_pos = [0.5*patch.get_width() + patch.get_xy()[0] for patch in rectlist[i/2]]
	plt.xticks(xticks_pos, xv, ha = "center", rotation=rotation)
	myutils.createDir(folder)
	filname = title.replace(' ','' )
	plt.savefig(folder+"/" + filname+".pdf")
	plt.close()


#Need to correct
def linegraph2(title, xlabel, ylabel, means, stds, folder):
	fig = plt.figure(figsize=(4, 3))
	xv1 = range(len(means))
	plt.errorbar([0.15, 1.15], means[0], yerr = stds[0], label="Upload Speed", fmt='s')
	plt.errorbar([0.3, 1.3], means[1], yerr = stds[1], label="Download Speed", fmt='o')
	#plt.scatter(xv1, means[0], marker="s")
	#plt.scatter(xv1, means[1], marker="o")

	#plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.title(title)

	plt.xticks([0.2, 1.2], xlabel, ha = "center")

	ax = plt.subplot(111)
	#ax.plot(marker="s")
	ax.legend()
	#plt.yticks(numpy.arange(0, 34, 5))
	fn = title + "Linegraph"
	fn = fn.replace(" ", "")
	fn = fn.replace("/","")
	gd = os.path.join(folder, "Linegraphs")
	myutils.createDir(gd)


	fn = os.path.join(gd, fn+".pdf")

	plt.savefig(fn)

	plt.close()

# Need to correct
def bargraph2(title, xlabel, ylabel, xv, yvss, folder):
	rectlist = []
	N = len(xv)
	ind = numpy.arange(N)
	i = 0
	colors = ['w','r', 'y', 'b', 'g','c']
	print yvss
	if len(yvss) == 0 or len(yvss[0]) == 0 or len(yvss[0][0]) ==0:
		return
	#print yvs, len(yvs)
	width = 8.0/(36)

	def autolabel(rects):
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2, 1.05*height, "", ha='center', va='bottom')

	fig, ax = plt.subplots(figsize = (8, 6))

	try:
		umeans = yvss[0][0]
		umedians = yvss[0][1]
		ustds = yvss[0][2]
		ufquartiles = yvss[0][3]
		utquartiles = yvss[0][4]
		udatasize = yvss[0][5]
		dmeans = yvss[1][0]
		dmedians = yvss[1][1]
		dstds = yvss[1][2]
		dfquartiles = yvss[1][3]
		dtquartiles = yvss[1][4]
		ddatasize = yvss[1][5]

	except:
		return

	#if "NDT upload and download speeds for Upstream Cross Traffic" in title:
	#print title
	#print umeans
	#print umedians
	#print ustds
	#print dmeans
	#print dmedians
	#print dstds


	legend = ['Upload(1st Quartile - Median)', 'Upload(3rd Quartile - Median)', 'Upload(Mean)', 'Downlaod(1st Quartile - Median)', 'Download(3rd Quartile - Median)', 'Download(Mean)']
	legend= ['Mean Upload Speed', 'Mean Download Speed']#, 'Rel. Data Size (act size/3)']
	rect3 = ax.bar(ind, umeans, width, color='#AF00AF', yerr = None)#ustds)
	mds = list(numpy.array(umedians) - numpy.array(ufquartiles))
	tqs = list(numpy.array(utquartiles) - numpy.array(umedians))

	rect1 = ax.bar(ind+width/3, mds, width/3, color = 'w', bottom=ufquartiles)
	rect2 = ax.bar(ind+width/3, tqs, width/3, color='w', bottom = umedians)
	#ind = ind + width
	rectlist.append(rect1)
	rectlist.append(rect2)
	rectlist.append(rect3)
	autolabel(rect1)
	autolabel(rect2)
	autolabel(rect3)


	ind = ind+width
	rect6 = ax.bar(ind, dmeans, width, color='#AFAF00', yerr=None)# = dstds)
	mds = list(numpy.array(dmedians) - numpy.array(dfquartiles))
	tqs = list(numpy.array(dtquartiles) - numpy.array(dmedians))

	rect4 = ax.bar(ind+width/3, mds, width/3, color = 'w', bottom=dfquartiles)
	rect5 = ax.bar(ind+width/3, tqs, width/3, color='w', bottom = dmedians)

	#rect7 = ax.bar(ind-width/12,ddatasize, width/6, color='r')

	#ind = ind + width
	rectlist.append(rect4)
	rectlist.append(rect5)
	rectlist.append(rect6)
	#rectlist.append(rect7)
	autolabel(rect4)
	autolabel(rect5)
	autolabel(rect6)
	#autolabel(rect7)
	ax.legend([rectlist[2], rectlist[5]], legend )
	#ax.set_title(title)
	xticks_pos = [0*patch.get_width() + patch.get_xy()[0] for patch in rect6]
	#ax.set_xticklabels(xv)
	plt.xticks(xticks_pos, xv, ha = "center")
	plt.yticks(numpy.arange(0, 34, 5))
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)

	folder = os.path.join(folder, "Barplots")
	createDir(folder)

	plt.savefig(folder+"/" + title+".pdf")
	plt.close()

def plotmatrix(matrix, title,folder, xlabels, xlabel='', ylabel='', norm=None):
	if norm=='log':
		norm = LogNorm()
	matrix = numpy.matrix(matrix)
	fig = plt.figure()
	fig, ax = plt.subplots(figsize=(10,8))
	ax.legend()
	ax.set_title(title)
	ax.set_aspect('equal')
	plt.imshow(matrix, interpolation='nearest', filternorm=0.6, cmap=cm.Greys, origin='lower', norm=norm)
	plt.colorbar()

	labels=range(7)
	plt.xticks(labels)
	ax.set_xticklabels(xlabels)
	ax.set_yticks(labels)
	ax.set_yticklabels(xlabels)
	plt.xlabel(xlabel)
	plt.grid = True
	plt.ylabel(ylabel)
	plt.savefig(folder+'/'+title+'.pdf')
	plt.close()

#Scatter plot

def scatterplot(scatterpoints,title,folder, xlabel='', ylabel='',labels=None,rotation='0',islog=False,stds=None):
	fig = plt.figure(figsize=(6, 4))
	xvs = []
	yvs = []
	counts = []
	for t in set(scatterpoints):
		x,y = t
		xvs.append(x)
		yvs.append(y)
		counts.append(scatterpoints.count(t))
	sizes = []
	for c in counts:
		sizes.append((float(c)*10))
	plt.scatter(xvs,yvs,s=sizes)
	'''
	xv1 = range(len(xv))
	xinc = 0.5/len(yvss)
	markers=['s', 'o', 'x', '^','v', '<', '>' ]
	for i in range(len(yvss)):
		marker = markers[i%len(markers)]
		xv1 = [ x + xinc for x in xv1]
		yvs = yvss[i]
		label = labels[i] if labels is not None else ''
		std = stds[i] if stds is not None else [0 for x in range(len(xv1))]
		plt.errorbar(xv1, yvs, yerr = stds, label=label, fmt=marker)
		#plt.errorbar([1,3], umeans, yerr = ustds, label="Upload Speed in Mbits/sec", fmt='o')
		params = {'legend.fontsize': 6,
		  'legend.linewidth': 2}
		plt.rcParams.update(params)
		#plt.scatter(xv1,yvs,)#, marker="s")
	'''
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.title(title)
	#plt.xticks(numpy.array([x+xinc*len(yvss)/2.0 for x in range(len(xv))]), xv, ha = "center", rotation=rotation)
	ax = plt.subplot(111)
	#ax.legend()
	#plt.yticks(numpy.arange(0, 34, 5))
	#plt.xscale('log')
	#plt.yscale('log')
	#fn = title + "Linegraph"
	#fn = fn.replace(" ", "")
	#fn = fn.replace("/","")
	#gd = os.path.join(folder)
	#myutils.createDir(gd)
	fn = title.replace(' ','')
	fn = os.path.join(folder, fn+".pdf")
	plt.savefig(fn)
	plt.close()


def scatterplots(scatterpoints,title,folder, xlabel='', ylabel='',legend = None,labels=None,rotation='0',islog=False,stds=None, xlimit=None, ylimit=None):
	fig, ax = plt.subplots(figsize = (5, 4))
	zvs = [z for (_,_,z) in scatterpoints]
	try:
		zvs = list(set(sorted(zvs)))
	except:
		pass
	markers=['o','o','*','s','D','*','D','1', 'x', '^','v', '<', '>' ]
	colors = ['w','r','r','k']
	asleg = []
	for i in range(len(zvs)):
		z = zvs[i]
		xvs = []
		yvs = []
		cs = []
		counts = []
		for t in set(scatterpoints):
			x,y,zt = t
			if zt != z:
				continue
			xvs.append(x)
			yvs.append(y)
			cs.append(colors[zvs.index(z)%len(colors)])
			counts.append(scatterpoints.count(t))
		sizes = []
		for c in counts:
			sizes.append(20)#(float(c)**2+10))
		marker = markers[zvs.index(z)%len(markers)]
		asleg.append(plt.scatter(xvs[0],yvs[0],s=[50], c = colors[zvs.index(z)%len(colors)], marker=marker))
		plt.scatter(xvs,yvs,s=sizes, c = cs, marker=marker)#, rasterized=True)
		#print z, marker, cs, len(sizes), sum(counts)
	#print 'plotted....'
	if not xlimit is None:
		ax.set_xlim(xlimit)
	if not ylimit is None:
		ax.set_ylim(ylimit)
	if legend is not None:
		ax.legend(asleg, legend, loc='upper right', fontsize=10)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.title(title)
	#plt.xticks(numpy.array([x+xinc*len(yvss)/2.0 for x in range(len(xv))]), xv, ha = "center", rotation=rotation)
	ax = plt.subplot(111)
	#ax.legend()
	#plt.yticks(numpy.arange(0, 34, 5))
	#plt.xscale('log')
	#plt.yscale('log')
	#fn = title + "Linegraph"
	#fn = fn.replace(" ", "")
	#fn = fn.replace("/","")
	#gd = os.path.join(folder)
	#myutils.createDir(gd)
	#plt.show()
	fn = title.replace(' ','')
	#fl = os.path.join(folder, fn+".pdf")
	#plt.savefig(fl)
	fn = os.path.join(folder, fn+".pdf")
	plt.savefig(fn)
	plt.close()

# specific to cellular routing project
def scatterplots1(scatterpoints,title,folder, xlabel='', ylabel='',labels=None,rotation='0',islog=False,stds=None, legend = None):
	fig, ax = plt.subplots(figsize = (5, 4))
	zvs = [z for (_,_,z) in scatterpoints]
	try:
		zvs = sorted(zvs)
	except:
		pass
	markers=['s','D','*']#,'D','1', 'x', '^','v', '<', '>' ]
	colors = ['r','b','k']
	print 'zvs:',set(zvs)
	mz = max(zvs)
	mz = 100
	zvs = [34,67,101]
	countixpisp = 0
	counttotal = 0
	asleg = []
	asleg.append(plt.scatter([0],[0],s=[50], c = 'c', marker='s'))
	for i in range(len(zvs)):
		z = zvs[i]
		pz = -1
		if i > 0:
			pz = zvs[i-1]
		xvs = []
		yvs = []
		cs = []
		counts = []
		for t in set(scatterpoints):
			x,y,zt = t
			if not(zt >= pz and zt < z):
				continue
			xvs.append(x)
			yvs.append(y)
			#cs.append('#'+str(z/mz *777777+111111).replace('.','')[:6])
			#if y >=33:
			#	cs.append(colors[0])
			#else:
			counts.append(scatterpoints.count(t))
			counttotal += scatterpoints.count(t)
			if x != 0:
				cs.append(colors[zvs.index(z)])
			else:
				cs.append('c')
			if y >= 33:
				countixpisp += scatterpoints.count(t)
		sizes = []
		for c in counts:
			sizes.append((float(c)**2+50))
		marker = markers[zvs.index(z)]
		asleg.append(plt.scatter(xvs[0],yvs[0],s=[50], c = cs[0], marker=marker))
		plt.scatter(xvs,yvs,s=sizes, c = cs, marker=marker)
		#print z, marker, cs[0]
	ax.set_xlim(-1,9)
	ax.set_ylim(-10,100)
	if legend is not None:
		ax.legend(asleg, legend, loc='upper right', fontsize=10)
	plt.plot([-1,9],[33,33],'r--')
	ax.text(0, 60, 'IXP-based ISPs (> 33% routes through IXP.)' , style='italic',fontsize=12)
	ax.text(0, 50, '('+str(countixpisp) + ' out of '+str(counttotal)+')' , style='italic',fontsize=12)

	'''
	xv1 = range(len(xv))
	xinc = 0.5/len(yvss)
	markers=['s', 'o', 'x', '^','v', '<', '>' ]
	for i in range(len(yvss)):
		marker = markers[i%len(markers)]
		xv1 = [ x + xinc for x in xv1]
		yvs = yvss[i]
		label = labels[i] if labels is not None else ''
		std = stds[i] if stds is not None else [0 for x in range(len(xv1))]
		plt.errorbar(xv1, yvs, yerr = stds, label=label, fmt=marker)
		#plt.errorbar([1,3], umeans, yerr = ustds, label="Upload Speed in Mbits/sec", fmt='o')
		params = {'legend.fontsize': 6,
		  'legend.linewidth': 2}
		plt.rcParams.update(params)
		#plt.scatter(xv1,yvs,)#, marker="s")
	'''
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.title(title)
	#plt.xticks(numpy.array([x+xinc*len(yvss)/2.0 for x in range(len(xv))]), xv, ha = "center", rotation=rotation)
	ax = plt.subplot(111)
	#ax.legend()
	#plt.yticks(numpy.arange(0, 34, 5))
	#plt.xscale('log')
	#plt.yscale('log')
	#fn = title + "Linegraph"
	#fn = fn.replace(" ", "")
	#fn = fn.replace("/","")
	#gd = os.path.join(folder)
	#myutils.createDir(gd)
	fn = title.replace(' ','')
	fn = os.path.join(folder, fn+".pdf")
	plt.savefig(fn)
	plt.close()


def scatterploto(xv, yvss,title,folder, xlabel='', ylabel='',labels=None,rotation='0',islog=False,stds=None):
	fig = plt.figure(figsize=(6, 4.5))
	xv1 = range(len(xv))
	xinc = 0.5/len(yvss)
	markers=['s', 'o', 'x', '^','v', '<', '>' ]
	for i in range(len(yvss)):
		marker = markers[i%len(markers)]
		xv1 = [ x + xinc for x in xv1]
		yvs = yvss[i]
		label = labels[i] if labels is not None else ''
		std = stds[i] if stds is not None else [0 for x in range(len(xv1))]
		plt.errorbar(xv1, yvs, yerr = stds, label=label, fmt=marker)
		#plt.errorbar([1,3], umeans, yerr = ustds, label="Upload Speed in Mbits/sec", fmt='o')
		params = {'legend.fontsize': 6,
		  'legend.linewidth': 2}
		plt.rcParams.update(params)
		#plt.scatter(xv1,yvs,)#, marker="s")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.xticks(numpy.array([x+xinc*len(yvss)/2.0 for x in range(len(xv))]), xv, ha = "center", rotation=rotation)
	ax = plt.subplot(111)
	ax.legend()
	#plt.yticks(numpy.arange(0, 34, 5))
	#plt.xscale('log')
	#plt.yscale('log')
	fn = title + "Linegraph"
	fn = fn.replace(" ", "")
	fn = fn.replace("/","")
	gd = os.path.join(folder, "Linegraphs")
	myutils.createDir(gd)
	fn = os.path.join(gd, fn+".pdf")
	plt.savefig(fn)
	plt.close()

def map2D(title, xlabel, ylabel, xv, yv, folder, same = False):

	if len(xv) is 0 or len(yv) is 0:
		return
	#print "xv: ", xv
	#print "yv: ", yv
	fig = plt.figure(figsize = (8,5))
	plt.plot(xv, yv,"ro")

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.xscale('log')
	#plt.yscale('log')
	#print min(xv), max(xv)
	#plt.axis([min(xv)/10, max(xv)*10, 0, (float(max(yv)))])
	fn = title
	xm = max(xv)
	ym = max(yv)
	if xm>ym:
		mx = xm
	else:
		mx = ym
	xm = min(xv)
	ym = min(yv)
	if xm>ym:
		mn = ym
	else:
		mn = xm

	fn = fn.replace(" ", "")
	fn = fn.replace("/","")
	gd = os.path.join(folder, "Map2D")
	myutils.createDir(gd)

	fn = os.path.join(gd, fn+".pdf")

	if same == True and xm is not None and ym is not None:
		plt.axis([mn,float(mx), mn, float(mx)])
	plt.savefig(fn)

	plt.close()



