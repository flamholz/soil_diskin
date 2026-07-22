!> Stub module for mo_kind
!> Provides the working precision kind used throughout JSBACH
MODULE mo_kind
  IMPLICIT NONE
  PUBLIC

  INTEGER, PARAMETER :: wp = selected_real_kind(12,307)

END MODULE mo_kind
