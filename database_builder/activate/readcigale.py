import astropy
import astropy.table, astropy.io
import numpy
import matplotlib.pyplot as plt
import sys

cigale_prefixes = ['attenuation', 'sfh', 'stellar', 'agn', 'dust', 'galaxy_mass', 'agn.fracAGN', 'universe']

# attenuation, sfh, stellar, agn, dust, galaxy_mass, agn.fracAGN ...
# universe.age, universe.lumdist, universe.redshift

def load_component(filename):
	print 'loading component %s' % filename
	tbl = astropy.table.Table.read('%s/computed_fluxes.txt' % filename)

	data = dict(params={}, fluxes={}, ID=None)

	freeparams = []
	patterns = {}
	aliases = {}

	for col in tbl.colnames:
		values = numpy.array(tbl[col])
		if col == 'id':
			data['ID'] = values
		elif any([col.startswith(prefix) for prefix in cigale_prefixes]):
			# parameter
			if numpy.min(values) == numpy.max(values):
				print '# fixed parameter: %s = %s' % (col, numpy.min(values))
			else:
				pattern = numpy.zeros(len(values))
				for i, v in enumerate(numpy.unique(values)):
					#print i, numpy.where(values == v) 
					pattern[numpy.where(values == v)] = i + 1
				othercols = [othercol for othercol, pat in patterns.iteritems() if (pat + pattern).std() == 0 or (pat == pattern).all()]
				patterns[col] = pattern
				if len(othercols) == 0:
					print 'variable parameter: %s = %s..%s' % (col, numpy.min(values), numpy.max(values))
					freeparams.append(col)
				else:
					print 'alias parameter: %s = %s..%s (alias of %s)' % (col, numpy.min(values), numpy.max(values), ','.join(othercols))
					aliases[col] = othercols[0]
				
				data['params'][col] = values
		else:
			# data
			data['fluxes'][col] = values

	print
	print 'free parameters:'
	N = 1
	for param in freeparams:
		Ni = len(numpy.unique(patterns[param]))
		print '   %3d x %s' % (Ni, param)
		N *= Ni

	print '*** number of combinations: %d' % N
	
	# load SEDs:
	seds = []
	for i in data['ID']:
		seds.append('%s/%s_best_model.xml' % (filename, i))
		#tbl_Fnu = astropy.table.Table.read('%s/%s_best_model.xml' % (filename, i), table_id='Fnu')
		#tbl_Flam = astropy.table.Table.read('%s/%s_best_model.xml' % (filename, i), table_id='Flambda')
		#seds.append((numpy.array(tbl_Fnu), numpy.array(tbl_Flam)))
	
	return dict(data=data, freeparams=freeparams, aliases=aliases, seds=seds)


components = []

for filename in sys.argv[1:]:
	components.append(load_component(filename))




