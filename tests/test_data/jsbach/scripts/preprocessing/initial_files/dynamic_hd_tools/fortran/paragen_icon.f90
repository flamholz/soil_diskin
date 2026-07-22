!> Contains the Parameter generation routines for the ICON grid.
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
MODULE paragen_icon
!
!     ******** Programm zur Generierung der Parameter fuer Overland- und
!              Riverflow. Da hierfuer jeweils die lineare Speicherkaskade
!              vorgesehen ist, werden die Koeffizienten n und k generiert.
!
!     ******** Revised version for 5-Min. resolution based on original
!              paragen.f Version 1.6 from Feb. 1997
!
!     ******** Programmierung und Entwicklung: Stefan Hagemann
!              Version 2.0 -- September 2014
!
!     ******** Version 3.0 - Module for ICON
!
!     ******** Version 3.1 - September 2015
!              Finale Setzen von ALF_N udn ARF_N auf ganze Zahlen
!              plus Korrektur von ALF_K und ARF_K zur Geschwindigkeitserhaltung
!              Dieses wurde vorher im HD model gemacht.
!
  IMPLICIT NONE


  INCLUDE 'netcdf.inc'
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12,307) !< double precission

  type hd_para
    REAL(dp) :: alf_k = 0._dp      ! Overland flow k
    REAL(dp) :: alf_n = 0._dp      ! Overland flow n
    REAL(dp) :: arf_k = 0._dp      ! Riverflow k
    REAL(dp) :: arf_n = 0._dp      ! Riverflow n
    REAL(dp) :: agf_k = 0._dp      ! Baseflow k
    REAL(dp) :: vellf = 0._dp      ! Overland flow velocity
    REAL(dp) :: velrf = 0._dp      ! Riverflow velocity
    REAL(dp) :: slope = 0._dp      ! Slope
    REAL(dp) :: area = 0._dp       ! Area
    REAL(dp) :: dis = 0._dp        ! Distance of flow in [m]
    REAL(dp) :: catarea = 0._dp    ! Catchment Area of gridbox
    REAL(dp) :: catnr = 0._dp      ! Number of gridboxes in catchment of gridbox
  end type hd_para
!
  type(hd_para), allocatable, dimension(:) :: para


  PUBLIC :: PARAGEN, TRACEAREA

CONTAINS

   SUBROUTINE PARAGEN(nicon, ticooro, ticodir, ticolon, ticolat)
!
!     ******** Variablenliste
!     ***
!     ***  nicon = Feldgroesse ICON
!     ***   AREA = Gitterbox-Flaechenarray [m^2]
!     ***   DLAT = Breitenabstand (konstant)[m]
!     ***   DLON = Array der Laengenabstaende [m]
!     ***   ticooro = Orographiearray [m]
!     ***   ticodir = Riverdirectionarray
!     ***  para(II)%alf_k = Array der Retentionskonstanten k - Overlandflow [day]
!     ***  para(II)%alf_n = Array der Speicherzahlen n       - Overlandflow
!     ***  para(II)%arf_k = Array der Retentionskonstanten k - Riverflow [day]
!     ***  para(II)%arf_n = Array der Speicherzahlen n - Riverflow
!     ***  para(II)%agf_k = Array der Retentionskonstanten k - Baseflow [day]
!     ***   FGSP = Initialisierungsarray fuer den linearen Baseflowspeicher
!     ***   FLAG = Landmasken-Array (= Landseemaske ohne Lakes)
!     ***  para(II)%slope = Slope = dh/dx
!     ***   ticowet = Wetland Percentage Array benutzt [%]
!     ***    NIS = Anzahl der Landgitterboxen mit ticoslinn = 0
!     ***  ticolake = Lake fraction [%]
!     ***
!     ***  para(II)%vellf = Velocity-Array fuer Overlandflow [m/s]
!     ***  para(II)%velrf = Velocity-Array fuer Riverflow [m/s]
!     ***
!     ***
!     ***  IFIN/IFOUT = Ein- bzw. Ausgabe-Format
!     ***          1 = Cray-Binaerformat
!     ***          2 = REGEN: Globales Binaerformat
!     ***          3 = REGEN: Waveiso2-Format
!     ***     LU = Logical Unit (50 fuer Cray, 20 fuer REGEN)
!     ***    LUF = Logical Unit fuer Flaechen-und Abstandsfile = 30
!     ***
!     ***  DNINP = Inputdateinamen, z.B. fuers globale Orographie-Array,
!     ***          Landmaske, Gletschermaske, globale Flaechen/Abstandsarray
!     ***          Riverdirectionfiles, Drainage Density Array
!     ***  DNOUT = Ausgabedatei (over_k.dat, over_n.dat, riv_k.dat, riv_n.dat)
!     ***
!     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
!     ***  IPARA = Art der Parameterisierung
!     ***          1 = Analog zu den Sausen-Koeffizienten mit reell
!     ***          8 = 1 & Inner Slope statt Slope f. Overlandflow
!     ***
!     ***  FK_LFK = Modifizierungsfaktor fuer k-Werte beim Overlandflow
!     ***  FK_LFN = Modifizierungsfaktor fuer n-Werte beim Overlandflow
!     ***  FK_RFK = Modifizierungsfaktor fuer k-Werte beim Riverflow
!     ***  FK_RFN = Modifizierungsfaktor fuer n-Werte beim Riverflow
!     ***  FK_GFK = Modifizierungsfaktor fuer k-Werte beim Baseflow
!     ***
!     ***  IBASE = Modell der Baseflowparameterisierung
!     ***           0 =   k = 300 days
!     ***           1 =   k = DX / 50 km * 300 days
!     ***           2 =   k = 300 days / Orographiefaktor
!     ***           3 =   k = DX / 50 km * 300 days / Orographiefaktor
!     ***           4 =   new ideas
!     ***
!     ***  ILAMOD = Modell der Lake-Dependance
!     ***           0 = No Lake-dep.
!     ***           1 = Charbonneau-Ansatz
!     ***           2 = tanh-Ansatz
!     ***           3 = linearer Ansatz
!     ***  VLA100 = Flow-Velocity bei 100 % Lake-Percentage [m/s]
!     ***  ISWMOD = Modell der Wetland-Dependance
!     ***           0 = No Swamp-dep.
!     ***           1 = Swamps werden den Lakes hinzuaddiert
!     ***           2 = tanh-Ansatz
!     ***           3 = linearer Ansatz
!     ***           4 = tanh nur fuer Overlandflow
!     ***           5 = tanh mit Wetlandtypes bei Riverflow
!     ***           6 = tanh mit Permafrost bei Riverflow
!     ***  VSW100 = Flow-Velocity bei 100 % Swamp-Percentage [m/s]
!     ***  PROARE = Area-Percentage, ab die Lake/Swamp-Percentage sich auswirkt
!     ***    VSAU = Minimum-Sausen-Velocity = principal dummy
!     ***
!
  USE mo_read_icon_trafo
!
      INTEGER, INTENT(in)        :: nicon
      REAL(dp), DIMENSION(nicon), INTENT(IN) :: ticooro, ticodir, ticolon, ticolat

      REAL(dp), DIMENSION(nicon) :: DLAT, DLON
      REAL(dp), DIMENSION(nicon) :: ticoslinn, ticostd
      REAL(dp), DIMENSION(nicon) :: ticolake, ticowet
      REAL(dp), DIMENSION(nicon) :: FLAG, ticooro_land

      REAL(dp), PARAMETER  :: C = 2._dp
      REAL(dp), PARAMETER  :: ALPHA = 0.1_dp
      REAL(dp), PARAMETER  ::   RERDE=6371000._dp
      REAL(dp), PARAMETER  ::   PI = 3.141592653589793_dp

      REAL(dp) :: PROARE, VLA100, VSW100, XIB, BB

      REAL(dp) :: FB, PIFAK, DH
      REAL(dp) :: VSAU, DXO, VSO, VDUM, ADUM, AD, CDUM
      CHARACTER*160 DNINP, ZEILE
      CHARACTER*160 DNORO,DDIR
      CHARACTER*6 :: CINI
      INTEGER  :: IPARA, IQUE, NDUM, ILAMOD
      ! INTEGER  :: IGMEM
      INTEGER  :: ISWMOD, IBASE, ISLOPE
      INTEGER  :: II, JJ, NIS, NWK1, NWK2, NN
      REAL(dp) :: FDUM, FK_LFK, FK_LFN, FK_RFK, FK_RFN, FK_GFK
!
!     ******* Basiswerte bzgl. Vindelaelven-Catchments:
!
      REAL(dp) :: ALF_K0 = 16.8522
      REAL(dp) :: ALF_N0 = 2.2214
!!!      REAL(dp) :: ALF_V0 = 0.0588
      REAL(dp) :: ALF_V0 = 1.0885
      REAL(dp) :: ALF_DX = 171000.

      REAL(dp) :: ARF_K0 = 0.4112
      REAL(dp) :: ARF_N0 = 9.1312
!!!      REAL(dp) :: ARF_V0 = 0.385
      REAL(dp) :: ARF_V0 = 1.0039
      REAL(dp) :: ARF_DX = 228000.
!
      INTEGER :: LUF = 30
      INTEGER :: ILOG = 1
!
!
      ALLOCATE(para(nicon))
!
!     *** Main directory with input files and input subdirectories
      CINI = "TDIRIN"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      DDIR=TRIM(ZEILE)  // "/"
!     ******* externe Belegungen aus Inputdatei PARAGEN.inp
      CINI = "IPARA"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IPARA = FLOOR(FDUM+0.01)
      CINI = "IQUE"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IQUE = FLOOR(FDUM+0.01)
!
!     *** Modifizierungsfaktoren der Parameter
      CINI = "FK_LFK"
      CALL PARINP(LUF, CINI, FK_LFK, ZEILE, IQUE)
      CINI = "FK_LFN"
      CALL PARINP(LUF, CINI, FK_LFN, ZEILE, IQUE)
      CINI = "FK_RFK"
      CALL PARINP(LUF, CINI, FK_RFK, ZEILE, IQUE)
      CINI = "FK_RFN"
      CALL PARINP(LUF, CINI, FK_RFN, ZEILE, IQUE)
      CINI = "FK_GFK"
      CALL PARINP(LUF, CINI, FK_GFK, ZEILE, IQUE)

!
!     ******* Input-Dateien-Auslese
      CINI = "TDNORO"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      DNORO=TRIM(DDIR) // TRIM(ZEILE)
      CALL read_netcdf_array(DNORO, 'cell_elevation', ticooro, nicon)

      CINI = "TDNARE"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      DNINP=TRIM(DDIR) // TRIM(ZEILE)
      CALL read_netcdf_array(DNINP, 'cell_area', para%area, nicon)

!     *** Use of inner slope (1) or normal slope (0) for Overland flow)
      CINI = "ISLOPE"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      ISLOPE = FLOOR(FDUM+0.01)
      IF (ISLOPE.EQ.1) THEN
        CINI = "TDNSLI"
        CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
        CALL read_netcdf_array(DNINP, 'orosig', ticoslinn, nicon)
      ENDIF
!
!     ********* Weiterverarbeitung der Inputarrays
!
!     *** Land mask
      DO II = 1, nicon
        IF (ticodir(II).GT.0.) THEN
           FLAG(II) = 1.
        ELSE
           FLAG(II) = 0.
        ENDIF
      ENDDO
!
!     *** Globale Flaechen und Abstaende
      PIFAK = PI/180._dp
      DO II = 1, nicon
        JJ = FLOOR(ticodir(II)+ 0.0001)
        IF (FLAG(II).GT.0.5) THEN
          FB = (ticolat(JJ) + ticolat(II)) / 2.
!
!         distance = DELphi [rad] * Earth radius [m]
          FDUM = ABS(ticolon(JJ) - ticolon(II))
          IF (FDUM.GT.300._dp) THEN                             ! Flow over date line
            FDUM = ABS(FDUM - 360_dp)
          ENDIF
          DLON(II) = FDUM * PIFAK * COS(FB*PIFAK)*RERDE
          DLAT(II) = ABS(ticolat(JJ) - ticolat(II)) * PIFAK * RERDE
          para(II)%dis = SQRT(DLAT(II)*DLAT(II) +  DLON(II)*DLON(II))
        ELSE
           DLON(II) = 0._dp
           DLAT(II) = 0._dp
        ENDIF
        IF (IQUE.NE.0.AND.(II.EQ.ILOG.OR.II.EQ.1)) THEN
           WRITE(*,*) 'Grid: ', II, ' Lon/lat=', ticolon(II), ticolat(II), 'FB=',FB, &
              ' dellon/dellat: ', ABS(ticolon(JJ) - ticolon(II)), ABS(ticolon(JJ) - ticolon(II))
           WRITE(*,*) '   at ',   COS( FB*PI/180. ), ' DL,DB: ', INT(DLON(II)), ' m', INT(DLAT(II)), ' m', &
                ' dis=', INT(para(II)%dis), ' m', ' area=', para(II)%area
        ENDIF
      ENDDO
!
!     *** Schreiben des globalen Laengen-Files
      DNINP = "area_dlat_dlon.txt"
      OPEN(LUF, FILE=DNINP, FORM='FORMATTED')
      WRITE(LUF,*) (para(II)%area,II=1,nicon)
      WRITE(LUF,*) (DLAT(II),II=1,nicon)
      WRITE(LUF,*) (DLON(II),II=1,nicon)
      CLOSE(LUF)
      WRITE(*,*) "*** Laengenfelder geschrieben in ",DNINP

! 126675.9
!     *** Orographie nur auf Landpunkten, 0 over ocean
      ticooro_land(:) = ticooro(:) * FLAG(:)
!
!     ******* Verzweigung nach Art der Parameterbestimmung
      IF (IPARA.NE.1 .AND. IPARA.NE.8) STOP 'IPARA does not exist'
!
!     ******* Parameterberechnung *****************************************
!
      NIS = 0
      DO II = 1, nicon
         IF (FLAG(II).LT.0.5) CYCLE
!
!        *** IL, IB = relative Richtungskoordinaten
!        *** Die 0.1-Summanden sind noetig wegen Cray-Rundungsungenauigkeiten
         JJ = FLOOR(ticodir(II)+ 0.0001)
!
!        *** Lokale Senke ?
         IF (JJ.EQ.II) CYCLE
!
         DH = ABS(ticooro_land(II) - ticooro_land(JJ))
         para(II)%slope = DH / para(II)%dis
         IF (para(II)%slope .GT. 1)THEN
            WRITE(*,*) 'Slope too large: ', para(II)%slope
            WRITE(*,*) '  DH: ', DH, ' Distance=',  para(II)%dis
            WRITE(*,*) '  II: ', II, ' oro=',  ticooro_land(II), '  JJ: ', JJ, ' oro=',  ticooro_land(JJ)
            !STOP
            para(II)%slope = 1.0
         ENDIF

         IF (ISLOPE.EQ.0) THEN
            ticoslinn(II) = para(II)%slope
            WRITE(*,*) "Inner slope replaced by slope to the next grid"
         ENDIF
!
!        *** Kommentar ?
         IF (IQUE.NE.0 .AND. II.EQ.ILOG) THEN
            WRITE(*,*) "II = ", II, "  ticodir = ",  &
                  ticodir(II), " ==> JJ = ", JJ
            WRITE(*,*) "DX = ", para(II)%dis, "  und DH =", DH
         ENDIF
!
!        *** Minimales V, das entspricht Minimum-Steigung
!        *** Ist nur Dummy, da DH = 0 nur bei lokalen Senken vorkommen darf.
         IF (DH.LE.0) THEN
!
!cc            VSAU = 0.01   (0.1 = 5.79 days auf 50 km)
            VSAU = 0.1
            WRITE(*,*) "Achtung: II = ",II, "  ticodir = ",   &
                 ticodir(II), "DX = ", para(II)%dis, "  und DH =", DH
         ELSE
            VSAU = C * (DH/para(II)%dis)**ALPHA
         ENDIF
!
!        ******** Sausen -Analogie
         IF (IPARA.EQ.1) THEN
!
!           *** Overlandflow
            para(II)%alf_k = ALF_K0 * para(II)%dis/ALF_DX * ALF_V0/VSAU
            para(II)%alf_n = ALF_N0
!
!           *** Riverflow
            para(II)%arf_k = ARF_K0 * para(II)%dis/ARF_DX * ARF_V0/VSAU
            para(II)%arf_n = ARF_N0
!
!        *** Sausen & Inner Slope statt Slope f. Overlandflow
         ELSE IF (IPARA.EQ.8) THEN
!
!           *** Overlandflow
            DXO = SQRT(DLAT(II)*DLAT(II) + DLON(II)*DLON(II))
            IF (ticoslinn(II).GT.0) THEN
               VSO = C * ticoslinn(II)**ALPHA
               para(II)%alf_k = ALF_K0 * DXO/ALF_DX * ALF_V0/VSO
            ELSE
!
!              *** Minimales V, das entspricht Minimum-Steigung
!              ***    ==> Inner Slope durch Normal Slope ersetzen
!
!cc            WRITE(*,*) "Achtung: JL = ",JL, "  JB =",JB, "  ticodir = "
!cc     &         ,  ticoslinn(II)
               para(II)%alf_k = ALF_K0 * para(II)%dis/ALF_DX * ALF_V0/VSAU
               NIS = NIS + 1
            ENDIF
            para(II)%alf_n = ALF_N0
!
!           *** Riverflow
            para(II)%arf_k = ARF_K0 * para(II)%dis/ARF_DX * ARF_V0/VSAU
            para(II)%arf_n = ARF_N0
!
!           *** Anwendung der Multiplikationsfaktoren aus Torneaelven-Experimenten
            para(II)%alf_k = para(II)%alf_k * 3.
            para(II)%alf_n = para(II)%alf_n * 0.5
            IF (para(II)%alf_n.LE.0.5) THEN
              NDUM = INT(para(II)%alf_n+ 0.5)
              para(II)%alf_k = para(II)%alf_k*para(II)%alf_n/NDUM
              para(II)%alf_n = NDUM
            ENDIF
!
            para(II)%arf_n = para(II)%arf_n * 0.6
            IF (para(II)%arf_n.LE.0.5) THEN
               NDUM = INT(para(II)%arf_n+ 0.5)
               para(II)%arf_k = para(II)%arf_k*para(II)%arf_n/NDUM
               para(II)%arf_n = NDUM
            ENDIF
!
         ENDIF
!
      ENDDO   ! nicon loop
!
      IF (NIS.NE.0) WRITE(*,*) NIS, " mal wurde alte Sausen-Analogie ",  &
           "mit Normal Slope statt Inner Slope verwendet."
!
!     *** ------------------------------------------------------------------
!     ******** PART 2: Korrekturen durch Lakes and Wetlands ****************
!     ********         sowie Baseflowparameterisierung      ****************
!     *** ------------------------------------------------------------------
!     ***
!     *** Belegen von ticolake mit Lake-Percentage [%]
!     *** Belegen von ticowet mit Wetland-Percentage [%]
!
!     ******* Input-Dateien-Auslese
!
!     *** Lake Percentage
      ! CINI = "TDNLAK"
      ! CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      ! CALL read_netcdf_array(DNINP, 'f_lakes', ticolake, nicon)
      ticolake(:) = 0.0
!
!     *** Wetland Percentage
      ! CINI = "TDNWET"
      ! CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
      ! CALL read_netcdf_array(DNINP, 'f_wet', ticowet, nicon)
      ticowet(:) = 0.0
!
!     *** Development-Parameter
      CINI = "ILAMOD"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      ILAMOD = FLOOR(FDUM+0.01)
      CINI = "VLA100"
      CALL PARINP(LUF, CINI, VLA100, ZEILE, IQUE)
!
      CINI = "ISWMOD"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      ISWMOD = FLOOR(FDUM+0.01)
      CINI = "VSW100"
      CALL PARINP(LUF, CINI, VSW100, ZEILE, IQUE)
!
      CINI = "PROARE"
      CALL PARINP(LUF, CINI, PROARE, ZEILE, IQUE)
!
!     *** Baseflowparameterisierung
      CINI = "IBASE"
      CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
      IBASE = FLOOR(FDUM+0.01)
!        CINI = 'IGMEM'
!        CALL PARINP(LUF, CINI, FDUM, ZEILE, IQUE)
!        IGMEM = FLOOR(FDUM+0.01)
!
!     *** Orographie-Streuung
      IF (IBASE.EQ.2 .OR. IBASE.EQ.3) THEN
        CINI = "TDNSIG"
        CALL PARINP(LUF, CINI, FDUM, DNINP, IQUE)
        CALL read_netcdf_array(DNINP, 'orostd', ticostd, nicon)
      ENDIF
!
!     ******* Parameter-Korrektur Schleife
!
      NWK1=0
      NWK2=0
      DO II = 1, nicon
         IF (FLAG(II).LT.0.5) CYCLE
!
!        *** IL, IB = relative Richtungskoordinaten
!        *** Die 0.1-Summanden sind noetig wegen Cray-Rundungsungenauigkeiten
         JJ = FLOOR(ticodir(II) + 0.001)
!
!        *** Lokale Senke ?
         IF (JJ.EQ.II) CYCLE
!
!        *** Anwendung der Multiplikationsfaktoren
         para(II)%alf_k = para(II)%alf_k * FK_LFK
         para(II)%alf_n = para(II)%alf_n * FK_LFN
         IF (para(II)%alf_n.LE.0.5) THEN
            NDUM = 1
            para(II)%alf_k = para(II)%alf_k*para(II)%alf_n/NDUM
            para(II)%alf_n = NDUM
         ENDIF
!
         para(II)%arf_k = para(II)%arf_k * FK_RFK
         para(II)%arf_n = para(II)%arf_n * FK_RFN
         IF (para(II)%arf_n.LE.0.5) THEN
            NDUM = 1
            para(II)%arf_k = para(II)%arf_k*para(II)%arf_n/NDUM
            para(II)%arf_n = NDUM
         ENDIF
!
!        ******* Lake Percentage
         IF (ticolake(II).GT.0) THEN
!
!          *** Riverflow
           VDUM = para(II)%dis / ( para(II)%arf_k*para(II)%arf_n*86400. )
!
!          * nach Charbonneau
           IF (ILAMOD.EQ.1) THEN
             IF (ticolake(II).GE.PROARE) THEN
               IF (VLA100.GE.VDUM)   &
                   WRITE(*,*) "II=", II,"  VDUM =", VDUM, "  Riv_k=", para(II)%arf_k
               ADUM =VDUM*(1 - (1 - VLA100/VDUM)**(100./ticolake(II)))
               para(II)%arf_n = 1.
               para(II)%arf_k = para(II)%dis / ( ADUM*86400. )
             ENDIF
!
!          * tanh-Ansatz
           ELSE IF (ILAMOD.EQ.2) THEN
             AD=(1-VLA100/VDUM)/2.
             ADUM = 1-AD *( tanh(4*PI *(ticolake(II)-PROARE)*0.01) +1)
             para(II)%arf_n = 1.
             para(II)%arf_k = para(II)%dis / ( ADUM*VDUM*86400. )

!cc             IF (ticolake(II).EQ.100)
!cc     &         WRITE(*,*) "JL=", JL, "  JB=", JB,
!cc     &         "  VDUM =", VDUM, "  ADUM=", ADUM,
!cc     &         "  k=", para(II)%arf_k
           ENDIF
!
!          ******* Overland flow
           VDUM = para(II)%dis / ( para(II)%alf_k*para(II)%alf_n*86400. )
!
!          * nach Charbonneau
           IF (ILAMOD.EQ.1) THEN
             IF (ticolake(II).GE.PROARE) THEN
             CDUM = VLA100/VDUM
               IF (VLA100.GE.VDUM) THEN
                 WRITE(*,*) "II=", II, "  VDUM =", VDUM, "  Over_k=", para(II)%alf_k
                 CDUM = 0.1
               ENDIF
               ADUM = VDUM* (1 - (1 - CDUM)**(100./ticolake(II)))
               para(II)%alf_n = 1.
               para(II)%alf_k = para(II)%dis / ( ADUM*86400. )
             ENDIF
!
!          * tanh-Ansatz
           ELSE IF (ILAMOD.EQ.2) THEN
!###             AD=(1-VLA100/VDUM)/2.
             AD=(1-VLA100*0.1/VDUM)/2.
!
             IF (VLA100*0.1.GE.VDUM) THEN
                 WRITE(*,*) "Lake: II=", II, "  VDUM =", VDUM, "  Over_k=", para(II)%alf_k
                 AD=(1-0.1)/2.
             ENDIF
             ADUM =1- AD *( tanh(4*PI *(ticolake(II)-PROARE)*0.01) +1)
             para(II)%alf_n = 1.
             para(II)%alf_k = para(II)%dis / ( ADUM*VDUM*86400. )
           ENDIF
         ENDIF
!
!        ******* Wetland Percentage *********************************
         IF (ISWMOD.GT.1 .AND. ticowet(II).GT.0) THEN
!
!          *** Riverflow
           VDUM = para(II)%dis / ( para(II)%arf_k*para(II)%arf_n*86400. )
!
!          *** Wetland type dependence (currently not implemented --> selected by ISWMOD
           IF (VSW100.LT.VDUM) THEN
!             *** 2 different types of wetlands
         IF (ISWMOD.EQ.5) THEN   ! Matthews type >= 6)
                AD=0.
             ELSE IF (ISWMOD.EQ.6) THEN
               AD=(1-VSW100/VDUM)/2.
           ENDIF
             IF (AD.GT.0.) THEN
               ADUM = 1-AD *(tanh(4*PI *(ticowet(II)-PROARE)*0.01) +1)
               para(II)%arf_n = 1.
               para(II)%arf_k = para(II)%dis / ( ADUM*VDUM*86400. )
             ENDIF
           ENDIF
!
!          *** Overland flow
           VDUM = para(II)%dis / ( para(II)%alf_k*para(II)%alf_n*86400. )
           IF (ISWMOD.EQ.5 .OR. ISWMOD.EQ.6) THEN
             IF (VSW100*0.1.LT.VDUM) THEN
                AD=(1-VSW100*0.1/VDUM)/2.
                NWK1=NWK1 + 1
             ELSE
                NWK2=NWK2 + 1
             ENDIF
             ADUM = 1-AD *( tanh(4*PI *(ticowet(II)-PROARE)*0.01) +1)
             para(II)%alf_n = 1.
             para(II)%alf_k = para(II)%dis / ( ADUM*VDUM*86400. )
           ELSE
              WRITE(*,*) "ERROR?: Wet: VDUM=", VDUM, " II=",II
           ENDIF
         ENDIF
!
!        *** Velocities [Lag in Days ==> Faktor 86400
         para(II)%vellf = para(II)%dis / ( para(II)%alf_k*para(II)%alf_n*86400. )
         para(II)%velrf = para(II)%dis / ( para(II)%arf_k*para(II)%arf_n*86400. )
!
!        ******* Baseflowparameterisierung
         IF (IBASE.EQ.0) THEN
!
!           *** Baseflow (vorlaeufig konstant - so wie bis Bebruar 1997)
            para(II)%agf_k = 300.
         ELSE IF (IBASE.EQ.1) THEN
!
!           *** Baseflow gitterbox-groessenabhaengig [m] / [m] * [day]
            para(II)%agf_k = para(II)%dis / 50000. * 300.
         ELSE IF (IBASE.EQ.2 .OR. IBASE.EQ.3) THEN
!
!           *** nach Beate Mueller
            BB = (ticostd(II) - 100.) / (ticostd(II) + 1000.)
            IF (BB.LT.0.01) BB=0.01
            XIB = 1. - BB + 0.01
            para(II)%agf_k = 300. / XIB
            IF (IBASE.EQ.3) para(II)%agf_k = para(II)%dis/50000. * para(II)%agf_k
         ELSE IF (IBASE.EQ.4) THEN
!
!           *** new ideas
            BB = (ticostd(II) - 100.) / (ticostd(II) + 1000.)
            IF (BB.LT.0.01) BB=0.01
            XIB = 1./(1. + 20*(SQRT(BB)-0.1))
            para(II)%agf_k = para(II)%dis/50000. * 300. * XIB
         ENDIF
!
         para(II)%agf_k = para(II)%agf_k * FK_GFK
!
!        *** Baseflowspeicherinitialisierung (vorlaeufig konstant ########)
!###         IF (IGMEM.EQ.1) FGSP(II) = 0.1 * AREA(JB) / 86400.
!
      ENDDO  ! loop ove rnicon
!
      WRITE(*,*) "Overlandflow: Wetland-Korrektur: VSW100=0.1 VSW100:"    &
              , "NWK1 =", NWK1
      WRITE(*,*) "Overlandflow: No Wet-Korrektur f. VSW100/10 > VDUM:"  &
               , "NWK2 =", NWK2
!
!     *** Check for N = whole number and correct K if necessary
      DO II = 1, nicon
        NN = NINT(para(II)%arf_n)
        IF (NN.GT.0) THEN
          para(II)%arf_k = para(II)%arf_k * para(II)%arf_n / NN
          para(II)%arf_n = REAL(NN, dp)
        ENDIF
        NN = NINT(para(II)%alf_n)
        IF (NN.GT.0) THEN
          para(II)%alf_k = para(II)%alf_k * para(II)%alf_n / NN
          para(II)%alf_n = REAL(NN, dp)
        ENDIF
      ENDDO
!
!     *** Adhoc returning for R2B9
      IF (nicon == 20971520) THEN
        DO II = 1, nicon
          IF (para(II)%alf_k > 10.0 ) THEN
            para(II)%alf_k = 2.4
          ENDIF
          IF (para(II)%arf_k > 0.04 ) THEN
            para(II)%arf_k = 0.0133
          ENDIF
        ENDDO
      END IF

!
!     *** The End
  END SUBROUTINE PARAGEN
!
!****************************************************************************
      SUBROUTINE PARINP(LU, CINI, FINI, ZEILE, IQUE)
!****************************************************************************
!
!     ******** Routine, welche das Auslesen von Initialisierungs-Parameter
!              zur Auswahl der Climagroessenberechnung aus der
!              Initialiesungsdatei PARAGEN.inp vornimmt.
!              Programmierung analog Routine METINP in METEOR.for.
!
!     ******** Version 1.0 - September 1995
!              Programmierung und Entwicklung: Stefan Hagemann
!
!     ***     LU = Logical Unit fuer Dateioeffnung
!     ***   CINI = Suchstring - Kennzeichnet den Parameternamen = 6 ZEICHEN
!     ***          If CINI(1:1) = "T" ==> Text gesucht.
!     ***   FINI = Parameterwert
!     ***  ZEILE = If Textvariable gesucht ==> Textinhalt
!     ***   IQUE = Kommentarvariable ( 0 = Kein Kommentar )
!
!     ********* Typischer Dateiaufbau der Initialisierungsdatei PARAGEN.inp
!     ***
!     ***     CINI1: Kommentar zu FINI1
!     ***     FINI1
!     ***     CINI2: Kommentar zu FINI2
!     ***     FINI2
!     ***       :            :
!     ***       :            :
!
      CHARACTER, INTENT(in)  :: CINI*6
      REAL(dp), INTENT(out)      :: FINI
      CHARACTER, INTENT(out) :: ZEILE*160
      INTEGER , INTENT(in)   :: LU, IQUE
      INTEGER :: IOS
!
!     *** Oeffnen der Initialisierungsdatei
      OPEN(LU,FILE="paragen.inp",ACCESS='sequential',FORM='formatted', &
           STATUS='OLD', IOSTAT=IOS)
      IF (IOS.NE.0) THEN
         WRITE(*,*) '*** ERROR opening file paragen.inp in PARINP'
         GOTO 999
      ENDIF
!
 100  READ(LU, '(A160)', END = 900) ZEILE
      IF (INDEX(ZEILE, CINI).NE.0) THEN
         IF (CINI(1:1).EQ."T") THEN
            READ(LU, '(A160)') ZEILE
            WRITE(*,*) "Initialisierung: ", CINI," = "
            WRITE(*,*) ZEILE
         ELSE
            READ(LU, *) FINI
            WRITE(*,*) "Initialisierung: ", CINI," = ", FINI
         ENDIF
         GOTO 999
      ENDIF
      GOTO 100
!
 900  WRITE(*,*) " "
      WRITE(*,*) "*** ", CINI," wurde nicht gefunden ***"
!
!     *** Schliessen der Datei LU
!
 999  CLOSE(LU, STATUS='KEEP',IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         WRITE(*,*) "******** Fehler bei Dateischliessung in PARINP"
      ENDIF
!
!        *** The End
  END SUBROUTINE PARINP
!
  SUBROUTINE TRACEAREA(nicon, ticodir, ticoslm, ticolon, ticolat, icocat)
!
!     ******** Variablenliste
  IMPLICIT NONE
  INTEGER, INTENT(in)        :: nicon
  REAL(dp), DIMENSION(nicon), INTENT(IN) :: ticodir, ticoslm, ticolon, ticolat
  INTEGER,  DIMENSION(nicon), INTENT(IN) :: icocat

  REAL(dp) :: ASUM
  REAL(dp), DIMENSION(nicon) :: ACAT
  INTEGER,  DIMENSION(nicon) :: iconr, imouth, irank
  INTEGER  :: II, JJ, NSUM, K, IR, IRMAX
  INTEGER, PARAMETER  :: LU = 10
  LOGICAL  :: LOGQUE
!
  iconr(:) = 0
  imouth(:) = 0
!
  DO II = 1, nicon
  IF (ticoslm(II).GT.0.5) THEN
    NSUM = 0
    ASUM = 0._dp
    JJ=II
    LOGQUE=.TRUE.
    DO WHILE(LOGQUE)
      IF (iconr(JJ).EQ.0) THEN
        NSUM = NSUM+1
        iconr(JJ) = NSUM
        ASUM = ASUM + para(JJ)%area
        para(JJ)%catarea = ASUM
      ELSE
        IF (NSUM.GT.0) THEN
          iconr(JJ) = iconr(JJ) + NSUM
          para(JJ)%catarea = para(JJ)%catarea + ASUM
        ENDIF
      ENDIF
      K = FLOOR(ticodir(JJ) + 0.001)
      IF (K.EQ.0) THEN
        WRITE(*,*) 'II = ', II, ' JJ= ', JJ, ' NR: ',iconr(JJ), ticoslm(JJ)
        STOP 'ERROR K = 0 in TRACEAREA'
      ENDIF
      IF (ticoslm(K).LT.0.5 .OR. K.EQ.JJ) THEN
        imouth(K) = icocat(JJ)
        iconr(K) = iconr(K) + NSUM
        para(K)%catarea = para(K)%catarea + ASUM
!!        WRITE(*, '( 2(F8.3,2X),I5, 2X, I7) ')   &
!!             ticolon(K), ticolat(K), imouth(K), iconr(K)
        LOGQUE=.FALSE.
      ENDIF
      JJ = K
    ENDDO
  ENDIF
  ENDDO
!
  para(:)%catnr = iconr(:)
!
  WRITE(*,*) 'Biggest catchment with ', MAXVAL(iconr(:)), 'gridboxes'
!
! *** Sort Catchments according to their size

    IRMAX=1
    DO II=1, nicon
      IF (imouth(II).GT.0) THEN
        IF (IRMAX.EQ.1) THEN
          ACAT(1) = para(II)%catarea
          irank(1) = II
        ELSE
          LOGQUE=.FALSE.
          DO IR=1, IRMAX
          IF (para(II)%catarea .GT. ACAT(IR)) THEN
            irank(IR+1:IRMAX+1) = irank(IR:IRMAX)
            ACAT(IR+1:IRMAX+1) = ACAT(IR:IRMAX)
            irank(IR) = II
            ACAT(IR) = para(II)%catarea
            LOGQUE=.TRUE.
            EXIT
          ENDIF
          ENDDO
        ENDIF
        IRMAX = IRMAX+1
        IF (.NOT.LOGQUE) irank(IRMAX) = II
      ENDIF
    ENDDO
!
! *** Oeffnen der Logdatei
  OPEN(LU,FILE="cat_icon.txt",ACCESS='sequential',FORM='formatted', &
           IOSTAT=K)
  IF (K.NE.0) THEN
    WRITE(*,*) '*** Fehler bei Dateioeffnung cat_icon.txt in TRACEAREA'
    STOP
  ENDIF
  DO IR=1, IRMAX
    II = irank(IR)
    IF (imouth(II).GT.0) THEN
      WRITE(LU, '( 2(F8.3,2X),I5, 2X, F10.2, 2X, I7, 2X, I1)')   &
         ticolon(II), ticolat(II), imouth(II), para(II)%catarea*1.E-6_dp, iconr(II), &
         FLOOR(ticoslm(II)+0.001)
    ENDIF
  ENDDO
  CLOSE(LU)
!
  END SUBROUTINE TRACEAREA


END MODULE paragen_icon

