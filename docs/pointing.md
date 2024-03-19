# Detector Pointing and Response

Working with detector pointing and

## Quaternions

TOAST includes support for a variety of quaternion operations.  The API largely follows the [quaternionarray]() package.  There are two common orderings of quaternion components.  TOAST follows the form which has the "angle" part at the end.  This matches the form used by scipy.XXXXXX.  The other form has the angle term at the beginning.  This is the form used for example by the Boost and SPT3G packages.  In the future TOAST will support both forms with a global package switch.

```{eval-rst}
.. automodule:: toast.qarray
    :members:
```

### Composition of Rotations

When combining rotations by multiplying quaternions, the order of multiplication depends on whether the rotations are relative to each other or are with respect to the same external coordinate system.


## Instrument Coordinates

In TOAST, the detector coordinate frame is defined to have the Z-axis pointed along the detector line of sight.  Similarly the X-axis in this frame should be aligned with the direction of the maximal polarization response.  For the most part, experiments using TOAST are free to define their telescope boresight pointing and focalplane coordinate frames however they like.  The critical point is that the final detector quaternion should rotate the coordinate system Z and X axes to align with the convention above.

TOAST comes with several built-in simulation operators for simulating boresight pointing and also simulating artificial focalplanes.  For these tools, we adopt a specific choice of instrument coordinate frames.  The focalplane coordinate frame is defined to be the one in which the detector positions are fixed (at least for some span of time).  The rotation from the overall coordinate system to focalplane coordinates is described by "boresight quaternions" at each time sample.  The (fixed) rotation from the focalplane boresight to the detector frame is given by one quaternion per detector.  

Since the Z-axis of the focalplane coordinate frame point "points out at the sky" and not back towards the observer, plotting the geometry from the observer's perspective can lead to confusion.  We define an auxilliary 2D focalplane coordinate axis that can be used for plotting this geometry.  This XiEtaGamma system is only used in TOAST for doing layout and plotting operations.  Internally, quaternions are used when building composite rotations.

## Simulated Boresight Pointing

For the `SimGround` operator [discussed here](ops:sim_ground), the boresight quaternions in the Az / El frame rotate the vector pointing at the zenith (the Z-axis) into the telescope line of sight.  They also rotate the vector pointing at zero azimuth (the X-axis) into a vector that is pointed "down" towards the direction of decreasing elevation.

**(figure here)**

For the `SimSatellite` operator [discussed here](ops:sim_satellite), the boresight quaternions rotate the Z and X axes of the ecliptic frame into the telescope line of sight and

## Stokes Weights (Detetor Response)

The polarized sky is often described in terms of the I, Q, U, and V Stokes parameters.
At each sample, a detector has some response to these Stokes parameters on the sky. For
a typical bolometric detector, this response can be modeled as a sequence of Mueller
matrices representing the elements in the optical path, followed by a total power
measurement (i.e. just using the first row of the resulting matrix).

### Example:  Linear Polarizer

It is useful to look at the simple example of a perfect linear polarizer to examine the conventions used.  We further assume that the circular polarization of the sky ("V" Stokes) is zero.  In this case the detector response to the sky signal depends only on the orientation of the polarizer with respect to the local meridian:

**(figure here showing Q/U axes on the sky and detector direction orientation)**

Recall that in the detector coordinate frame, the line of sight is along the transformed
Z-axis and the polarization orientation is along the transformed X-axis. The Q and U
response depends on the angle between the polarization orientation and the local
meridian of the coordinate axes. The right-handed rotation that takes the meridian
vector to the polarization orientation vector (which are co-planar) is:

$$
\begin{eqnarray}
\vec{d} & = & \text{direction vector} \\
\vec{p} & = & \text{polarization orientation} \\
\vec{m} & = & \text{meridian vector} \\
a & = & \arctan{\left(\frac{(\vec{m} \times \vec{p}) \cdot \vec{d}}{\vec{m} \cdot \vec{p}}\right)}
\end{eqnarray}
$$

The meridian vector is orthogonal to the direction vector, and we can express its components in terms of the components of that vector:

$$
\begin{eqnarray}
\vec{d} & = & d_x\vec{i}, \; d_y\vec{j}, \; d_z\vec{k} \\
{d_r}^2 & = & {d_x}^2 + {d_y}^2 \\
\vec{m} & = & \frac{1}{d_r} \left( -d_x d_z \vec{i}, \; -d_y d_z \vec{j}, \; {d_r}^2 \vec{k} \right)
\end{eqnarray}
$$


### Basic Operator

The included `StokesWeights` operator provides a simple model of detector response that is useful for many generic simulation use cases.  Specific experiments will almost certainly want to implement a more complicated operator that includes additional or time-varying Mueller matrix elements in the optics chain.  This operator includes support for partial linear polarization and also a perfect rotating or stepped half-wave plate.

```{eval-rst}
.. autoclass:: toast.ops.StokesWeights
    :members:
```

$$
d = cal \left[\frac{(1+eps)}{2} I + \frac{(1-eps)}{2} \left[Q \cos{2a} + U \sin{2a}\right]\right]
$$

# Sky Coordinates and Pixelization


## Horizon Coordinates


## Celestial Coordinates

- Equatorial

- Ecliptic

- Gallactic


## Pixelization

TOAST includes built-in support for Healpix pixelization as well as some types of WCS projections.  These are implemented in the `PixelsHealpix` and `PixelsWCS` operators:

```{eval-rst}
.. autoclass:: toast.ops.PixelsHealpix
    :members:
```

```{eval-rst}
.. autoclass:: toast.ops.PixelsWCS
    :members:
```

If you just need support for a different flat projection, you may find it easier to
modify the existing `PixelsWCS` operator. Custom pixelization operators need to be able
to create a `PixelDistribution` object (see
[this section of the data model](pixel:dist)).





# Customizations

It is possible to implement a completely custom pointing model and use that with TOAST
analysis tools. For this to work, you need to implement:

1.  An Operator or other code that loads or simulates boresight pointing quaternions
    using whatever coordinate conventions you like. These should be stored in a shared
    data object in each Observation.

2.  A detector pointing operator that can produce final detector quaternions from the
    boresight pointing. If your experiment uses the same focalplane coordinate frame as
    the TOAST defaults, then you can use the standard `PointingDetectorSimple` operator
    for this step.

3.  An operator that returns the Stokes weights for each detector at each sample. If you
    have a very simple optical model (partial linear polarizer with perfect half wave
    plate), then the included `StokesWeights` operator may suffice.
