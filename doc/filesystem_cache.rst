.. _filesystem-cache:

Relevant classes
================

- auto_disk_array (src/python/cache.py)

    - derives from numpy.ndarray with some custom overloads

    - `__new__`

        - used for object creation instead of `__init__`

    - `__array_finalize__`

        - used to finalize the object created from `__new__`

    - `__del__`

        - important to the functionality of this class
        - it checks the number of `auto_disk_array`'s in existance according to the Cache class and if the destructor recognizes it is the last instance, it saves a copy to disk (e.g creates an FsBuffer object), add the FsBuffer object to the Cache's dictionary, tells the Cache  object to destroy the reference in memory and deletes the dictionary entry for the auto_disk_array in Cache

    - `__array_wrap__`

        - used to ensure operations return an auto_disk_array instead of the base class

    - `__setitem__`

        - needed to update the auto_refs count in Cache when auto_disk_array is involved in a ufuncs operation

    - `__array_ufunc__`

        - needed to ensure that methods involving auto_disk_array and other numpy arrays return types of auto_disk_array

- FsBuffer (src/python/cache.py)

    - A simple class that creates a temporary file and handles reading and writing numpy arrays from and to disk.
    - the location of the temporary files directory is set either directly with assigning the filepath to `Cache.toast_cache_tmpdir` or with then environment variable `TOAST_TMPDIR`, where the former overrides the latter, in the event both are set.

- Cache (src/python/cache.py)

    - a previously existing class with extensions for above
    - new dictionary of FsBuffer objects

        - Cache will check the dictionary of FsBuffer objects when querying for existance by default and if there is an entry it will report that the reference exists.
        - If a reference is requested from Cache and the object is an FsBuffer object, it will load the object into memory, delete the FsBuffer object, and return an auto_disk_array
        - enabling the usage of FsBuffer and auto_disk_array are never enabled by default. One must either create the array using `use_disk=True` as a parameter or use one of the Cache member methods: `set_use_disk(name, use_disk=True)` or `move_to_disk(name)`
        - to disable the usage of disk of an object that previously was using disk, use the Cache member functions: `set_use_disk(name, use_disk=False)` or `load_from_disk(name)`

    - new dictionary of auto_disk_array counting (`_auto_refs`)

        - the numpy array still exists in _refs
        - however, the presence of the name in `_auto_refs` means that references will return `auto_disk_array` objects which can be treated exactly like a numpy ndarray, the difference being that when all instances of `auto_disk_array` have been garbage collected, auto_disk_array will save the underlying numpy ndarray to disk and free the memory

    - The member function `clear` has a new parameter to control whether one should destroy the disk references in addition to the memory references. By default, this is enabled
    - The member function `free` has the functionality of `clear` without the disk references being cleared
    - The member function `create` has a new parameter to control whether or not the array being created should use the `FsBuffer`/`auto_disk_array` paradigm instead of always keeping the array in memory. This parameter is `use_disk` and, by default, this parameter is `use_disk=False`
    - The member function `destroy` has an additional paramter: `remove_disk`. By default, this parameter is `remove_disk=True` but can be set to `False` to ensure that no cached file copies are destroyed.
    - The member function `exists` has a new parameter: `use_disk`. By default, this parameter is `use_disk=True`. This parameter controls whether to consider FsBuffer objects when returning whether an object exists or does not or when getting a reference to that object
    - There is a new member method `copy` which returns a copy of the numpy array which is free to be modified without the changes being reflected by the Cache reference
    - There is a new member method `auto_reference_count` that makes it easy+safe to check the number of `auto_disk_array` instances for a given name, if any exist at all
    - The member method keys was extended to include the FsBuffer dictionary keys since, in general, the user should treat any numpy ndarray using `FsBuffer`+`auto_disk_array` paradigm the same as any other numpy ndarray reference in Cache

Unit testing
=============

- test_fscache (src/python/tests/cache.py)

    - Create and store numpy arrays (using the filesystem cache) of the following types:

        - float (64-bit)
        - float (32-bit)
        - integer (signed, 64-bit)
        - integer (unsigned, 64-bit)
        - integer (signed, 32-bit)
        - integer (unsigned, 32-bit)
        - integer (signed, 16-bit)
        - integer (unsigned, 16-bit)
        - integer (signed, 8-bit)
        - integer (unsigned, 8-bit)

    - Check that `Cache.create(...)` routine return a type of `auto_disk_array`
    - Store that `auto_disk_array` object in a dictionary
    - Check that only two instances of `auto_disk_array` matching the given name exist

        - one returned Cache, one in dictionary, none in Cache

    - delete the returned `auto_disk_array` instance
    - Start a new loop and check only one instance of `auto_disk_array` exists (one in dictionary)
    - Ask cache for a reference -- variable name == `data`
    - Check two instances of `auto_disk_array` exist
    - Check both instances of `auto_disk_array` are of type `auto_disk_array`
    - Modify the `auto_disk_array` instance with the variable name `data`
    - Check that `auto_disk_array` instance with the variable name `data` and the dictionary instance are still the same after modifying `data` instance
    - Delete the instance named `data`
    - Modify the `auto_disk_array` instance in dictionary
    - Ask the Cache object for a reference again and compare with dictionary instance to ensure they are equal
    - Delete all the dictionary instances so that there are no more instances of `auto_disk_array`
    - Check the cache that a numpy array for the given ID name exists (with `use_disk=True`)

        - checking that when all `auto_disk_array` instances were garbage collected, the array was put into FsBuffer object (i.e. stored on disk)

    - Check the Cache object that a numpy array for the given ID name does not exist in memory (with `use_disk=False`)

        - checking again that when all `auto_disk_array` instances were garbage collected, the array was put into FsBuffer object (i.e. stored on disk)

    - Check that the `auto_disk_array` count is equal to zero at this point

        - `self.assertTrue(self.cache.auto_reference_count(name) == 0)`

    - Call `Cache.free()` (similar to clear but not deleting FsBuffer objects)
    - Ensure FsBuffer object still exist, no objects in memory, and auto_refs has a count == 0
    - Call `Cache.destroy(name, remove_disk=False)`
    - Ensure FsBuffer objects still exist, no objects in memory, and auto_refs has a count == 0
    - Clear the cache (also destroying FsBuffer objects)
    - Ensure no FsBuffer objects exist, no objects in memory, and auto_refs ahs a count == 0

- test_fscache_copy (src/python/tests/cache.py)

    - define function for getting copies

        - create a numpy arrays using filesystem cache (i.e. `use_disk=True`)
        - ask Cache object for copies (via `Cache.copy(name)`)
        - check that objects returned by copy method are NOT `auto_disk_array` types
        - store the copied objects in dictionary
        - modify the copy objects

    - check that Cache has numpy arrays stored as FsBuffer objects
    - ask Cache for a reference
    - check that modified copy object does not equal reference object
    - destroy the cache and the clones

- test_fscache_size (src/python/tests/cache.py)

    - THIS UNIT TEST IS A BIT PRONE TO FAILURE DUE TO PYTHON DOING ANNOYING GARBAGE COLLECTION STUFF

        - For example, it has been noted that in Python 3.6.4 that calling `print(...)` with a numpy array or writing to a file will hold onto a reference, causing the numpy array to not be garbage collected, even when `del` is called on the object that was printed/written

    - This unit test uses the `timemory.rss_usage` class to check that Cache is, in fact, freeing memory and storing the results in disk

        - Despite the occasional failures, which have been attempted to be worked around, this test is necessary

    - These are the steps of the unit test

        1. Measure base RSS (resident set size) usage
        2. Create a 5000x5000 64-bit float numpy array in memory (~200 MB)
        3. Record RSS 'regular-reference' (after deleting local reference so only reference is in Cache class)
        4. Move the array to disk (i.e. move to FsBuffer object)
        5. Record RSS 'move-to-disk'
        6. Move the array from disk back to memory
        7. Record RSS 'load-from-disk'
        8. Clear the cache
        9. Record RSS 'clear-cache'
        10. Create another 5000x5000 64-bit float numpy array (~200 MB)
        11. Record RSS 'auto-disk-array-init'
        12. Delete `auto_disk_array` object (should move numpy array to FsBuffer)
        13. Record RSS 'auto-disk-array-del'
        14. Clear the cache again
        15. Record RSS 'final'

    - Define a `relative_difference` function for handling small/negligible differences in memory

    - These are the RSS checks and their expected results

        - 'regular-reference' RSS should be greater than 'move-to-disk' RSS since the array should be on disk
        - 'load-from-disk' RSS should be greater than 'auto-disk-array-del' RSS since the array should be on disk
        - 'auto-disk-array-del' RSS should be greater than 'auto-disk-array-init' RSS since the array should be on disk
        - The RSS difference should be negligible between:
        - 'regular-reference' and 'load-from-disk'
        - 'move-to-disk' and 'auto-disk-array-del'
        - 'load-from-disk' and 'auto-disk-array-init'
        - 'clear-cache' and 'final'
