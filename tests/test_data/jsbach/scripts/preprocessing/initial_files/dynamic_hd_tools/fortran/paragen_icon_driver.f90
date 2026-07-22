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
  PROGRAM paragen_icon_driver

  USE mo_read_icon_trafo
  USE paragen_icon, ONLY: para, paragen, tracearea, PARINP
  USE netcdf

  IMPLICIT NONE
!
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12,307) !< double precision

  INTEGER :: K
  INTEGER :: ISLM                 ! Choice of land sea mask variable: 1=slm, 2=cell_sea_land_mask
  INTEGER  :: nicon
  REAL(dp), ALLOCATABLE :: ticodir(:), ticolon(:), ticolat(:)
  INTEGER, ALLOCATABLE :: icocat(:)            ! Catchments on ICON grid
!
  INTEGER, ALLOCATABLE :: iconbor(:,:)          ! ICON neighbors (nicon,3) != to what ncdump -h shows!!!!
  INTEGER, ALLOCATABLE  :: icoextnbor(:,:)      ! Extended ICON neighbors -> 12 boxes
  REAL(dp), ALLOCATABLE :: ticoslm(:)           ! ICON land sea mask
  REAL(dp), ALLOCATABLE :: ticooro(:)           ! ICON Orography
!
! *** Variables for calling Routine PARINP
  CHARACTER  :: CINI*6
  REAL(dp)   :: FDUM
  CHARACTER  :: ZEILE*160
  INTEGER    :: LUF=30
  INTEGER    :: IQUE
!
  REAL(dp), PARAMETER :: PI=3.14159265358979
  CHARACTER DDIR*240, DNOUT*240, ORDER*200
  CHARACTER*240 DNAM, DNORO, DNCELL, DNFRAC, DNFDIR
  INTEGER,  DIMENSION(2) :: nreg
  INTEGER :: ncid,cell_dimid,ticodir_varid,agf_k_varid
  INTEGER :: alf_k_varid,alf_n_varid,arf_k_varid,arf_n_varid
  INTEGER :: flon_varid,flat_varid
!
! *** Input files and directory
!
! *** Main directory with input files and input subdirectories
  CINI = "TDIRIN"
  CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
  DDIR=TRIM(ZEILE)  // "/"
  ! *** Smoothed orography, e.g. r2b4/oro_otto.nc
  CINI = "TDNOTO"
  CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
  DNORO=TRIM(DDIR) // TRIM(ZEILE)
! *** file with neighbor_cell_index(3)
  CINI = "TDNARE"
  CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
  DNCELL=TRIM(DDIR) // TRIM(ZEILE)
! *** file with land sea mask and choice of land sea mask variable
  CINI = "ISLMVA"
    CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
    ISLM = FLOOR(FDUM+0.01)
  CINI = "TDNSLM"
    CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
    DNFRAC=TRIM(DDIR) // TRIM(ZEILE)
! *** file with river directions
  CINI = "TDNFDIR"
  CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
  DNFDIR = TRIM(DDIR) // TRIM(ZEILE)

  DNOUT="icon_fdir_temp.nc"

! *** Read number of icon cells from orography file
  CALL read_netcdf_dims(DNCELL, nreg, nicon)

! *** ICON grid
  ALLOCATE(ticodir(nicon))
  ALLOCATE(ticolon(nicon))
  ALLOCATE(ticolat(nicon))
  ALLOCATE(icocat(nicon))
  ALLOCATE(iconbor(nicon,3))
  ALLOCATE(icoextnbor(nicon,12))
  ALLOCATE(ticooro(nicon))
  ALLOCATE(ticoslm(nicon))
!
! *** Read ICON info
  WRITE (*,*) "Reading in Data Files"
  CALL read_netcdf_array(DNCELL, 'lon_cell_centre', ticolon, nicon)
  CALL read_netcdf_array(DNCELL, 'lat_cell_centre', ticolat, nicon)
  CALL read_netcdf_array(DNORO, 'cell_elevation', ticooro, nicon)
  CALL read_netcdf_array(DNFDIR, 'next_cell_index', ticodir, nicon)
  ticolon(:) = ticolon(:)/PI*180._dp
  ticolat(:) = ticolat(:)/PI*180._dp

  IF (ISLM.EQ.1) THEN
    CALL read_netcdf_array(DNFRAC, 'slm', ticoslm, nicon)
  ELSE IF (ISLM.EQ.2) THEN
    CALL read_netcdf_array(DNFRAC, 'cell_sea_land_mask', ticoslm, nicon)
  ELSE
    STOP 'ISLM not defined'
  ENDIF
  CALL read_netcdf_intarray(DNCELL, 'neighbor_cell_index', iconbor, 3, nicon)
!
! *** Generate 12 neighbors
  WRITE (*,*) "Generating Neighbors"
  CALL GENNBORS(nicon, iconbor, icoextnbor)
!
! *** Parameter generation
  WRITE (*,*) "Running Parameter Generation"
  CALL PARAGEN(nicon, ticooro, ticodir, ticolon, ticolat)
!
! *** Write files
!
! ******* Open der binaeren Outputdatei
  call check_return_code(nf90_create(DNOUT,nf90_netcdf4,ncid))
  call check_return_code(nf90_def_dim(ncid,"cell",nicon,cell_dimid))
  call check_return_code(nf90_def_var(ncid,"FDIR", &
                                      nf90_double,cell_dimid,ticodir_varid))
  call check_return_code(nf90_def_var(ncid,"ALF_K", &
                                      nf90_double,cell_dimid,alf_k_varid))
  call check_return_code(nf90_def_var(ncid,"ALF_N", &
                                      nf90_double,cell_dimid,alf_n_varid))
  call check_return_code(nf90_def_var(ncid,"ARF_K", &
                                      nf90_double,cell_dimid,arf_k_varid))
  call check_return_code(nf90_def_var(ncid,"ARF_N", &
                                      nf90_double,cell_dimid,arf_n_varid))
  call check_return_code(nf90_def_var(ncid,"AGF_K", &
                                      nf90_double,cell_dimid,agf_k_varid))
  call check_return_code(nf90_def_var(ncid,"FLON", &
                                      nf90_double,cell_dimid,flon_varid))
  call check_return_code(nf90_def_var(ncid,"FLAT", &
                                      nf90_double,cell_dimid,flat_varid))
  call check_return_code(nf90_enddef(ncid))
  call check_return_code(nf90_put_var(ncid,ticodir_varid,ticodir))
  call check_return_code(nf90_put_var(ncid,alf_k_varid,para(:)%alf_k))
  call check_return_code(nf90_put_var(ncid,alf_n_varid,para(:)%alf_n))
  call check_return_code(nf90_put_var(ncid,arf_k_varid,para(:)%arf_k))
  call check_return_code(nf90_put_var(ncid,arf_n_varid,para(:)%arf_n))
  call check_return_code(nf90_put_var(ncid,agf_k_varid,para(:)%agf_k))
  call check_return_code(nf90_put_var(ncid,flon_varid,ticolon))
  call check_return_code(nf90_put_var(ncid,flat_varid,ticolat))
  call check_return_code(nf90_close(ncid))
!
  ORDER="cdo -f nc4 setgrid," // TRIM(DNCELL) // " " // TRIM(DNOUT) // " fdir_icon.nc"
  CALL  EXECUTE_COMMAND_LINE(ORDER)
  STOP

CONTAINS

 subroutine check_return_code(return_code)
   integer, intent(in) :: return_code
     if(return_code /= nf90_noerr) then
       print *,trim(nf90_strerror(return_code))
       stop
     end if
 end subroutine check_return_code

END PROGRAM paragen_icon_driver

!-------------------------------------------------------------------------
  SUBROUTINE GENNBORS(nicon, iconbor, icoextnbor)
!-------------------------------------------------------------------------
  IMPLICIT NONE
  INTEGER, INTENT(in)  :: nicon
  INTEGER, DIMENSION(nicon, 3),  INTENT(in) :: iconbor ! ICON neighbors
  INTEGER, DIMENSION(nicon, 12),  INTENT(out) :: icoextnbor ! Extended ICON neighbors -> 12 boxes

  INTEGER :: II, JJ, K, KN, KNR, JNB, KN2, KNB, K2
  INTEGER, DIMENSION(6)    :: IDUM, IDUM2
  INTEGER, DIMENSION(3)    :: I3
!
  icoextnbor(:, 1:3) = iconbor(:, 1:3)
!
  DO II=1, nicon
!
!   *** 6 neighbors of the direct 3 neighbors
    KNR=3
    DO KN=1, 3
      JJ = iconbor(II, KN)
      DO K=1, 3
        IF (iconbor(JJ, K).NE.II) THEN
           KNR = KNR+1
           icoextnbor(II, KNR) = iconbor(JJ, K)
           IDUM(KNR-3) = iconbor(JJ, K)
        ENDIF
      ENDDO
    ENDDO
    IDUM2(:) = IDUM(:)
!
!   *** final 3 neighbors
    DO KN=1, 6
      JJ = IDUM(KN)
      DO K=1,3
        JNB =  iconbor(JJ, K)
        I3(:)=iconbor(II,:)
        WHERE (I3.EQ.JNB)
          I3=1
        ELSEWHERE
          I3=0
        END WHERE
        IF (SUM(I3(:)).EQ.0) THEN
        DO K2=1,3
          KNB =  iconbor(JNB, K2)
          DO KN2=1, 6
            IF (KNB.EQ.IDUM2(KN2) .AND. KN.NE.KN2) THEN
              KNR = KNR+1
              icoextnbor(II, KNR) = JNB
              IF (II.EQ.1) WRITE(*,*) KN,'. ', IDUM(KN), JNB, '->', KN2, '. ', IDUM2(KN2)
              IDUM2(KN) = 0 ; IDUM2(KN2)=0
              EXIT
            ENDIF
          ENDDO
        ENDDO
        ENDIF
      ENDDO      ! Loop over the 6
    ENDDO
    IF (II.EQ.1) WRITE(*,*) "II=1, KNR=", KNR, ' --> ', icoextnbor(II, :)
  ENDDO
!
!
  KNR = 0
  DO II=1, nicon
    DO KN=1, 12
      IF (icoextnbor(II, KN).EQ.0) THEN
         IF (KN-1.EQ.0) STOP 'GENNBORS error'
         icoextnbor(II, KN) = icoextnbor(II,KN-1)
         KNR = KNR + 1
      ENDIF
    ENDDO
  ENDDO
!
  WRITE(*,*) 'Zero neighbors: ', KNR, ' --> corrected'
!
  END SUBROUTINE GENNBORS
!
