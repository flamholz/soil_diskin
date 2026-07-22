!> Module containing subroutines to read data from an ICON grid
!>
!> ICON-Land
!>
!> ---------------------------------------
!> Copyright (C) 2013-2024, MPI-M, MPI-BGC
!>
!> Contact: icon-model.org
!> Authors: AUTHORS.md
!> See LICENSES/ for license information
!> SPDX-License-Identifier: BSD-3-Clause
!> ---------------------------------------
!>
!!   Contains the read subroutines for the ICON grid.
!!
!! Following a module by Leonidas Linardakis, MPIM, 2015
!!
!!
MODULE mo_read_icon_trafo

!   USE mo_kind,               ONLY: wp

  IMPLICIT NONE

  PRIVATE

  INCLUDE 'netcdf.inc'
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12,307) !< double precission


  PUBLIC :: read_netcdf_dims, read_netcdf_array, read_netcdf_realarray, read_netcdf_intarray
  !--------------------------------------------------------------------

CONTAINS

  !--------------------------------------------------------------------
  SUBROUTINE nf(return_status)
    INTEGER, INTENT(in) :: return_status

    IF (return_status /= nf_noerr) THEN
      WRITE(0,*)'mo_io_grid netCDF error', nf_strerror(return_status)
      STOP
    ENDIF

  END SUBROUTINE nf
  !-------------------------------------------------------------------------

  !-------------------------------------------------------------------------
  SUBROUTINE read_netcdf_dims(file_name, nreg, nicon)
  !-------------------------------------------------------------------------
    CHARACTER(LEN=*), INTENT(in) :: file_name

    INTEGER :: ncid, varid, dimid
    INTEGER,  DIMENSION(2), INTENT(out)  :: nreg
    INTEGER,  INTENT(out)  :: nicon
    REAL(dp), DIMENSION(2) :: f2

    WRITE(0,*) 'Read Dimensions from ICON file ', TRIM(file_name)
    !-------------------------------------------------------------------------

    CALL nf(nf_open(TRIM(file_name), nf_nowrite, ncid))
!    CALL nf(nf_inq_dimid(ncid, 'src_grid_dims', dimid))
!    CALL nf(nf_inq_dimlen(ncid, dimid, nicon))

    ! CALL nf(nf_inq_varid(ncid,'src_grid_dims' , varid))
    ! CALL nf(nf_get_var_int(ncid, varid, nicon))

    ! CALL nf(nf_inq_varid(ncid,'dst_grid_dims' , varid))
    ! CALL nf(nf_get_var_double(ncid, varid, f2))
    ! nreg(:) = INT(f2(:)+0.001)
    CALL nf(nf_inq_dimid(ncid, 'cell', dimid))
    CALL nf(nf_inq_dimlen(ncid, dimid, nicon))

    CALL nf(nf_close(ncid))

  END SUBROUTINE read_netcdf_dims

  !-------------------------------------------------------------------------
  SUBROUTINE read_netcdf_array(file_name, carray, fdat, ndim)
  !-------------------------------------------------------------------------
    CHARACTER(LEN=*), INTENT(in) :: file_name
    CHARACTER(LEN=*), INTENT(in) :: carray

    INTEGER :: ncid,  varid
    INTEGER,  INTENT(in)  :: ndim
    REAL(dp), DIMENSION(ndim) :: fdat

    WRITE(0,*) 'Read Array from ICON file ', TRIM(file_name)
    !-------------------------------------------------------------------------
    CALL nf(nf_open(TRIM(file_name), nf_nowrite, ncid))

    CALL nf(nf_inq_varid(ncid, carray, varid))
    CALL nf(nf_get_var_double(ncid, varid, fdat(:)))

    CALL nf(nf_close(ncid))

  END SUBROUTINE read_netcdf_array

  !-------------------------------------------------------------------------
  SUBROUTINE read_netcdf_realarray(file_name, carray, fdat, ndim)
  !-------------------------------------------------------------------------
    CHARACTER(LEN=*), INTENT(in) :: file_name
    CHARACTER(LEN=*), INTENT(in) :: carray

    INTEGER :: ncid,  varid
    INTEGER,  INTENT(in)  :: ndim
    REAL, DIMENSION(ndim) :: fdat

    WRITE(0,*) 'Read Array from ICON file ', TRIM(file_name)
    !-------------------------------------------------------------------------
    CALL nf(nf_open(TRIM(file_name), nf_nowrite, ncid))

    CALL nf(nf_inq_varid(ncid, carray, varid))
    CALL nf(nf_get_var_real(ncid, varid, fdat(:)))

    CALL nf(nf_close(ncid))

  END SUBROUTINE read_netcdf_realarray

  !-------------------------------------------------------------------------
  SUBROUTINE read_netcdf_intarray(file_name, carray, fdat, ndim1, ndim2)
  !-------------------------------------------------------------------------
    CHARACTER(LEN=*), INTENT(in) :: file_name
    CHARACTER(LEN=*), INTENT(in) :: carray

    INTEGER :: ncid,  varid
    INTEGER,  INTENT(in)  :: ndim1
    INTEGER,  INTENT(in)  :: ndim2
!!    INTEGER, DIMENSION(ndim1, ndim2) :: fdat
    INTEGER fdat(ndim1, ndim2)

    WRITE(0,*) 'Read Array from ICON file ', TRIM(file_name)
    !-------------------------------------------------------------------------
    CALL nf(nf_open(TRIM(file_name), nf_nowrite, ncid))

    CALL nf(nf_inq_varid(ncid, carray, varid))
    CALL nf(nf_get_var(ncid, varid, fdat))
!!    CALL nf(nf_get_var_int(ncid, varid, fdat))

    CALL nf(nf_close(ncid))

  END SUBROUTINE read_netcdf_intarray


END MODULE mo_read_icon_trafo
