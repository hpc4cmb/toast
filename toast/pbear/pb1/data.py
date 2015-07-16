# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter
from itertools import groupby

import re

import numpy as np

import tidas as td

import PyArcReader as pyarc
import AnalysisBackend.misc.parse_flat_map as parsemap
import AnalysisBackend.indexing.featureflags_chile as features


def tidas_type ( flags ):
    ret = 0
    arctype = pyarc.typeString ( flags )
    if ( ( arctype == "BOOL" ) or ( arctype == "UCHAR" ) ):
        ret = "uint8"
    elif ( arctype == "CHAR" ):
        ret = "int8"
    elif ( arctype == "SHORT" ):
        ret = "int16"
    elif ( arctype == "USHORT" ):
        ret = "uint16"
    elif ( arctype == "INT" ):
        ret = "int32"
    elif ( arctype == "UINT" ):
        ret = "uint32"
    elif ( arctype == "FLOAT" ):
        ret = "float32"
    elif ( arctype == "COMPLEX" ):
        ret = "float64"
    elif ( arctype == "DOUBLE" ):
        ret = "float64"
    else:
        raise RuntimeError ( 'Unknown PyArcReader datatype' )
    return ret


def get_target ( val ):
    comp = 0
    ret = 0
    if ( val & features.F_SCIENCE ):
        comp += 1
        ret = features.F_SCIENCE
    if ( val & features.F_RADIOPOINTING ):
        comp += 1
        ret = features.F_RADIOPOINTING
    if ( val & features.F_OPTICALPOINTING ):
        comp += 1
        ret = features.F_OPTICALPOINTING
    if ( val & features.F_ARRAYMAP ):
        comp += 1
        ret = features.F_ARRAYMAP
    if ( val & features.F_PIXELMAP ):
        comp += 1
        ret = features.F_PIXELMAP
    if ( val & features.F_AZTILT ):
        comp += 1
        ret = features.F_AZTILT
    if ( val & features.F_STIMFLUXCAL ):
        comp += 1
        ret = features.F_STIMFLUXCAL
    if ( val & features.F_STIMPOLCAL ):
        comp += 1
        ret = features.F_STIMPOLCAL
    if ( val & features.F_TUNEPERF ):
        comp += 1
        ret = features.F_TUNEPERF
    if ( val & features.F_ELEVATION_BALANCE ):
        comp += 1
        ret = features.F_ELEVATION_BALANCE
    if ( comp > 1 ):
        print >> sys.stderr, "WARNING:  duplicate observation targets (features = {})".format(val)
    return ret


def get_scan ( val ):
    comp = 0
    ret = 0
    if ( val & features.F_CES ):
        comp += 1
        ret = features.F_CES
    if ( val & features.F_TRACKRASTER ):
        comp += 1
        ret = features.F_TRACKRASTER
    if ( val & features.F_FIXEDTRACK ):
        comp += 1
        ret = features.F_FIXEDTRACK
    if ( val & features.F_ELNOD ):
        comp += 1
        ret = features.F_ELNOD
    if ( val & features.F_SLEWING ):
        comp += 1
        ret = features.F_SLEWING
    if ( comp > 1 ):
        print >> sys.stderr, "WARNING:  duplicate scan types (features = {})".format( val )
    return ret


def get_config ( val ):
    ret = features.F_STIMCHOP | features.F_STIMPOLON | features.F_STIMPOLSPIN | features.F_GUNNON | features.F_GUNNCHOP | features.F_TUNING | features.F_GUNNPOLCAL | features.F_DSCPOLCAL
    return ret & val


def get_state ( val ):
    ret = {}
    ret['obs'] = ( val & features.F_ANALYZE )
    ret['target'] = get_target(val)
    ret['scan'] = get_scan(val)
    ret['conf'] = get_config(val)
    return ret


def np2string ( data ):
    temp = np.copy(data)
    temp = temp[np.nonzero(temp)]
    temp = ''.join(temp.view('S{}'.format(len(temp))))
    temp = re.sub(r"[\(\)]", "", temp)
    return temp


def feature_split ( flags ):

    transitions = [ i for (i, (b, e)) in enumerate( zip( flags[:-1], flags[1:] ) ) if ( b != e ) ]

    in_obs = False

    obs = []
    curobs = {}
    curobs[ 'scans' ] = []
    curscan = {}
    curscan[ 'configs' ] = []
    curconf = {}

    if ( flags[0] & features.F_ANALYZE ):
        in_obs = True
        curobs[ 'first' ] = 0
        curobs[ 'target' ] = get_target ( flags[0] )
        curscan[ 'first' ] = 0
        curscan[ 'type' ] = get_scan ( flags[0] )
        curconf[ 'first' ] = 0
        curconf[ 'bits' ] = get_config ( flags[0] )
        print "entering obs at {}".format( curobs['first'] )
        print "  entering scan type {} at {}".format( curscan['type'], curscan['first'] )
        print "    entering config {} at {}".format( curconf['bits'], curconf['first'] )
    else:
        in_obs = False

    for trans in transitions:

        if ( ( flags[ trans ] & features.F_ANALYZE ) and ( not ( flags[ trans+1 ] & features.F_ANALYZE ) ) ):
            # we are leaving an observation
            in_obs = False
            curobs[ 'last' ] = trans
            curscan[ 'last' ] = trans
            curconf[ 'last' ] = trans
            print "    leaving config {} at {}".format( curconf['bits'], curconf['last'] )
            curscan[ 'configs' ] += [ curconf ]
            print "  leaving scan type {} at {}".format( curscan['type'], curscan['last'] )
            curobs[ 'scans' ] += [ curscan ]
            print "leaving obs at {}".format( curobs['last'] )
            obs += [ curobs ]
            curobs = {}
            curobs[ 'scans' ] = []
            curscan = {}
            curscan[ 'configs' ] = []
            curconf = {}
        elif ( ( not ( flags[ trans ] & features.F_ANALYZE ) ) and ( flags[ trans+1 ] & features.F_ANALYZE ) ):
            # we are entering a new observation
            in_obs = True
            curobs[ 'first' ] = trans+1
            curobs[ 'target' ] = get_target ( flags[ trans+1 ] )

            curscan[ 'first' ] = trans+1
            curscan[ 'type' ] = get_scan ( flags[ trans+1 ] )
            
            curconf[ 'first' ] = trans+1
            curconf[ 'bits' ] = get_config ( flags[ trans+1 ] )
            print "entering obs at {}".format( curobs['first'] )
            print "  entering scan type {} at {}".format( curscan['type'], curscan['first'] )
            print "    entering config {} at {}".format( curconf['bits'], curconf['first'] )

        if ( in_obs ):

            oldtype = get_scan ( flags[ trans ] )
            newtype = get_scan ( flags[ trans+1 ] )

            oldconf = get_config ( flags[ trans ] )
            newconf = get_config ( flags[ trans+1 ] )

            if ( oldtype != newtype ):
                # we are entering a new scan type
                curconf[ 'last' ] = trans
                print "    leaving config {} at {}".format( curconf['bits'], curconf['last'] )
                curscan[ 'configs' ] += [ curconf ]
                curconf = {}
                curscan[ 'last' ] = trans
                print "  leaving scan type {} at {}".format( curscan['type'], curscan['last'] )
                curobs[ 'scans' ] += [ curscan ]
                curscan = {}
                curscan[ 'configs' ] = []
                curscan[ 'first' ] = trans+1
                curscan[ 'type' ] = newtype
                curconf[ 'first' ] = trans+1
                curconf[ 'bits' ] = newconf
                print "  entering scan type {} at {}".format( curscan['type'], curscan['first'] )
                print "    entering config {} at {}".format( curconf['bits'], curconf['first'] )
            elif ( oldconf != newconf ):
                # we are entering a new configuration
                curconf[ 'last' ] = trans
                print "    leaving config {} at {}".format( curconf['bits'], curconf['last'] )
                curscan[ 'configs' ] += [ curconf ]
                curconf = {}
                curconf[ 'first' ] = trans+1
                curconf[ 'bits' ] = newconf
                print "    entering config {} at {}".format( curconf['bits'], curconf['first'] )

    if in_obs:
        curconf[ 'last' ] = flags.shape[0] - 1
        print "    leaving config {} at {}".format( curconf['bits'], curconf['last'] )
        curscan[ 'last' ] = flags.shape[0] - 1
        curscan[ 'configs' ] += [ curconf ]
        print "  leaving scan type {} at {}".format( curscan['type'], curscan['last'] )
        curobs[ 'last' ] = flags.shape[0] - 1
        curobs[ 'scans' ] += [ curscan ]
        print "leaving obs (target = {}) at {}".format( curobs['target'], curobs['last'] )
        obs += [ curobs ]

    return obs


def build_schema ( af ):
    # get the field list
    fields = af.fields()
    # sort fields by SPF
    spf = {}
    for key, val in fields.iteritems():
        ( mapname, board, block, col ) = af.regname ( key )
        if val['spf'] not in spf:
            spf[val['spf']] = []
        spf[val['spf']] += [ key ]
    # organize schema by SPF
    schm = {}
    lookup = {}
    for key, val in spf.iteritems():
        gname = "spf_{}".format(key)
        schm[gname] = {}
        for f in val:
            schm[gname][f] = (tidas_type(fields[f]['type']), "")
            lookup[f] = key
    return (schm, lookup)


class Volume ( object ):

    def __init__ ( self, path ):
        self.path = path
        self.vol = td.Volume(path, backend="hdf5", comp="gzip", mode="w")


    def __del__(self):
        self.vol.close()


    def append ( self, af, year, month, day, hwmap=None ):
        xmlmap = None
        #if ( hwmap is not None ):
        #    xmlmap = parsemap.build_index_maps( hwmap )
        nframes = af.frames()
        schemas, lookup = build_schema(af)
        data = af.read( [ 'array-frame-features', ], 0, nframes-1 )
        obs = feature_split(data['array-frame-features'])
        # get the source name and field name for each observation.
        # We look up this value in the middle of each observation,
        # to avoid any funny edge effects.
        for ob in obs:
            mid = (ob['first'] + ob['last'])/2
            data = af.read(['antenna0-tracker-source','antenna0-tracker-field_name'], mid, mid)
            str_source = np2string(data['antenna0-tracker-source'])
            str_field = np2string(data['antenna0-tracker-field_name'])
            print str_source
            print str_field
            ob['source'] = 'src_' + str_source
            ob['field'] = 'fld_' + str_field

        # determine the state of observations at the current end
        # of the specified day and source and field.
        br = self.vol.root()
        if year not in br.block_names():
            br.block_add(year)
        by = br.block_get(year)
        if month not in by.block_names():
            by.block_add(month)
        bm = by.block_get(month)
        if day not in bm.block_names():
            bm.block_add(day)
        bd = bm.block_get(day)

        newobs = False

        for ob in obs:
            if ob['source'] not in bd.block_names():
                bd.block_add(ob['source'])
                newobs = True
            bsrc = bd.block_get(ob['source'])
            if ob['field'] not in bsrc.block_names():
                bsrc.block_add(ob['field'])
                newobs = True
            bfld = bsrc.block_get(ob['field'])
            bnames = sorted(bfld.block_names())
            if len(bnames) == 0:
                newobs = True
            if not newobs:
                # on first iteration, check if we are continuing
                # an existing observation
                if len(bnames) > 0:
                    latest = bfld.block_get(bnames[-1])
                    g = latest.group_get("spf_1")
                    fval = g.read('array-frame-features', g.size-1, 1)
                    state = get_state(fval)
                    if ( ( not state['obs']) or (state['target'] != ob['target'])):
                        newobs = True
            if newobs:
                bnm = "obs_{}".format(len(bnames))
                print "Starting new block {}".format(bnm)
                bfld.block_add(bnm)
                b = bfld.block_get(bnm)
                # initialize schema
                for gname in sorted(schemas.keys()):
                    g = td.Group(schema=schemas[gname])
                    b.group_add(gname, g)
                offset = 0
            else:
                bnm = bnames[-1]
                print "Continuing block {}".format(bnm)
                b = bfld.block_get(bnm)
                g = b.group_get("spf_1")
                offset = g.size

            # now write all data to the groups

            print "writing data at frame offset {}".format(offset)

            data = af.read(lookup.keys(), 0, nframes-1)
            for fname in lookup.keys():
                gname = "spf_{}".format(lookup[fname]) 
                g = b.group_get(gname)
                foffset = offset * lookup[fname]
                fwrite = data[fname].shape[0]
                #print "write field {} samples {} - {}".format(fname, foffset, foffset+fwrite-1)



            # all future observations in the list are new...
            newobs = True



class Archive ( object ):
    """
    Class which represents a single archive file.

    Each Pointing class has one or more detectors, and this class
    provides pointing quaternions and flags for each detector.

    Args:
        path (string): the path to the archive file.
    """

    def __init__ ( self, path ):

        self.fieldsep = "-"

        self.arc = None

        self.path = path

        # Create register map dictionary
        self.open()
        self.meta = pyarc.archiveFileToDictionary ( self.arc, 1 )

        # Get number of frames
        self.nframes = self.meta[ "frameCount" ]

        # Get array map revision
        self.revision = self.meta[ "arrayMapRevision" ]

        # ignore these bolometer fields, since they are redundant
        # with the hardware map.
        boloignore = [ "xy", "id", "fpIndex" ]

        # Build field list and register tree

        self.regs = {}
        self.fieldlist = {}

        for regmap in self.meta[ "registerMaps" ]:
            mapname = self.meta[ "registerMaps" ][ regmap ][ "name" ]
            boards = {}
            for board in self.meta[ "registerMaps" ][ regmap ][ "boards" ]:
                boardname = self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "name" ]
                blocks = {}
                for block in self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "blocks" ]:
                    blockname = self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "blocks" ][block][ "name" ]
                    blockinfo = {}

                    if ( not ( ( boardname == "bolometers" ) and blockname in boloignore ) ):
                    
                        typeflags = int ( self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "blocks" ][block][ "flags" ] )
                        blockinfo[ 'type' ] = typeflags

                        naxes = int ( self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "blocks" ][block][ "axesCount" ] )

                        dims = self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "blocks" ][block][ "axesLength" ]

                        fieldname = mapname + self.fieldsep + boardname + self.fieldsep + blockname

                        if ( naxes < 2 ):
                            blockinfo[ 'fields' ] = 1
                            spf = int ( self.meta[ "registerMaps" ][ regmap ][ "boards" ][ board ][ "blocks" ][block][ "elementCount" ] )
                            blockinfo[ 'spf' ] = spf
                            self.fieldlist[ fieldname ] = {
                                'name' : fieldname,
                                'spf' : spf,
                                'type' : typeflags
                            }
                        else:
                            subfields = int ( dims[0] )
                            spf = int ( dims[1] )
                            blockinfo[ 'fields' ] = subfields
                            blockinfo[ 'spf' ] = spf
                            for sf in np.arange( subfields ):
                                sfname = "%s%s%d" % ( fieldname, self.fieldsep, sf )
                                self.fieldlist[ sfname ] = {
                                    'name' : sfname,
                                    'spf' : spf,
                                    'type' : typeflags
                                }

                        if ( blockinfo[ 'spf' ] > 200 ):
                            print "WARNING: block {} has spf of {}".format(blockname, blockinfo[ 'spf' ])

                        blocks[ blockname ] = blockinfo

                boards[ boardname ] = blocks

            self.regs[ mapname ] = boards



    def close ( self ):
        """
        Explicitly close the archive file in the underlying 
        PbArchive library.
        """
        if ( self.arc != None ):
            status = pyarc.closeArchiveFile ( self.arc )
            if ( status != 0 ):
                msg = "destructor failed to close {}".format( self.path )
                raise IOError ( msg )
            self.arc = None


    def open ( self ):
        """
        Open the file, prior to doing some operation.  Explicit open/close
        of the underlying PbArchive file (rather than doing this in the 
        constructor/destructor) is necessary due to flakiness and instability
        in the C library.
        """
        status = 0
        ( status, self.arc ) = pyarc.openArchiveFile ( self.path )
        if ( status != 0 ):
            msg = "Constructor failed to open {}".format( self.path )
            raise IOError ( msg )


    def meta ( self ):
        """
        Return the register map dictionary from the underlying 
        PyArcReader module.
        """
        return self.meta


    def registers ( self ):
        """
        Return the register list as a hierarchy of dictionaries 
        based on map / board / block names.
        """
        return self.regs


    def fields ( self ):
        """
        Return a dictionary with the fieldnames as keys where each 
        item is a dictionary with members "name", "type" (string 
        representing archive file data type), and "spf" (samples per frame).
        """
        return self.fieldlist


    def frames ( self ):
        """
        Return the total number of frames in the archive file.
        """
        return self.nframes


    def rev ( self ):
        """
        Return the arrayMapRevision variable from the archive 
        file header.
        """
        return self.revision


    def regname ( self, fieldname ):
        """
        Given a field name, return the Register Map, Board, 
        Block and Column as a tuple.

        @type fieldname: string
        @param fieldname: Field name to parse.
        """
        f = fieldname.split ( self.fieldsep )
        if ( len ( f ) > 3 ):
            return ( f[0], f[1], f[2], int(f[3]) )
        else:
            return ( f[0], f[1], f[2], 0 )


    def read ( self, fields, first_frame, last_frame ):
        """
        Read a whole number of frames worth of samples from 
        all fields in the given list.  Fields from the same 
        register block are coalesced into the minimal number 
        of requests.  Data is reshaped before return.

        @type fields: list
        @param fields: A list of field names to read.
        @type first_frame: integer
        @param first_frame: The first data frame to read.
        @type last_frame: integer
        @param last_frame: The last data frame to read (inclusive).

        @rtype: dictionary
        @return: A dictionary of numpy arrays, one entry per field.
        """

        # get a unique list of fields
        seen = set()
        ufields = [ x for x in fields if x not in seen and not seen.add(x)]

        # make a list of all requests
        blockcolumns = {}
        blockspf = {}

        for field in ufields:
            if ( field not in self.fieldlist ):
                msg = "field {} does not exist in {}".format( field, self.path )
                raise IOError ( msg )
            ( mapname, board, block, col ) = self.regname ( field )

            blockpath = mapname + self.fieldsep + board + self.fieldsep + block

            blockspf[ blockpath ] = self.fieldlist[ field ][ 'spf' ]

            if blockpath in blockcolumns:
                blockcolumns[ blockpath ] += [ int(col) ]
            else:
                blockcolumns[ blockpath ] = [ int(col) ]

        # coalesce column requests and build read list
        blockranges = {}
        blockrangeindx = {}
        readspec = []
        readindx = 0

        for blockpath, cols in blockcolumns.iteritems():
            ( mapname, board, block, emptycol ) = self.regname ( blockpath )
            colsort = sorted( cols )

            ranges = []
            for k, g in groupby( enumerate(colsort), lambda (i,x):i-x ):
                group = map(itemgetter(1), g)
                ranges.append((group[0], group[-1]))

            blockranges[ blockpath ] = ranges

            blockrangeindx[ blockpath ] = []
            for r in ranges:
                readspec += [ ( self.arc, mapname, board, block, r[0], r[1], 0, blockspf[blockpath]-1, 0, 0, first_frame, last_frame ) ]
                blockrangeindx[ blockpath ].append( readindx )
                readindx += 1

        blockrangeset = []
        blockrangevec = []

        for field in ufields:
            ( mapname, board, block, col ) = self.regname ( field )
            blockpath = mapname + self.fieldsep + board + self.fieldsep + block
            ranges = blockranges[ blockpath ]

            i = 0
            for r in ranges:
                if ( ( int(col) >= r[0] ) and ( int(col) <= r[1] ) ):
                    blockrangeset.append ( i )
                    blockrangevec.append ( int(col) - r[0] )
                i += 1

        # READ
        
        status, data = pyarc.readRegisters( readspec )
        if ( status != 0 ):
            msg = "failed read on {}".format( self.path )
            raise IOError ( msg )
        
        output = {}

        # reshape data
        for field in zip ( ufields, blockrangeset, blockrangevec ):
            ( mapname, board, block, col ) = self.regname ( field[0] )
            blockpath = mapname + self.fieldsep + board + self.fieldsep + block

            spec = blockrangeindx[ blockpath ][ field[1] ]
            rg = blockranges[ blockpath ][ field[1] ]
            nvecs = rg[1] - rg[0] + 1

            #print "field %s is vector %d of request %d" % ( field[0], field[2], spec )

            output[ field[0] ] = data[ spec ][ field[2] ][:][:]
            frames = output[ field[0] ].shape[0]
            spf = output[ field[0] ].shape[1]

            #print "  reshaping %d x %d --> %d" % ( frames, spf, frames * spf )
            output[ field[0] ].shape = ( frames * spf )

        return output


if __name__ == "__main__":

    import sys
    afile = sys.argv[1]
    year = sys.argv[2]
    month = sys.argv[3]
    day = sys.argv[4]

    af = Archive(afile)

    v = Volume("temp_tidas")
    v.append(af, year, month, day)

    af.close()

