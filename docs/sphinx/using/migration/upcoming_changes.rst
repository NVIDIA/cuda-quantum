:orphan:

Upcoming changes
================================================

ABI change: ``kraus_op::data`` is always double precision
----------------------------------------------------------

``cudaq::kraus_op::data`` is now unconditionally
``std::vector<std::complex<double>>`` regardless of the simulation
precision, and the ``kraus_op::precision`` member has been removed.
Constructing a ``kraus_op`` from single-precision data still works (the
values are converted to double); generic code should use
``kraus_op::value_type`` to refer to the element type. ``fp32`` backends
continue to compute in single precision.

Upcoming changes to ``sample`` and ``observe``
------------------------------------------------

Details about the planned changes and migration guidance are coming soon.
