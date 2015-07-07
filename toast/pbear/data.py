# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import PyArcReader as pyarc


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
