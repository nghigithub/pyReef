!!~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~!!
!!                                                                                   !!
!!  This file forms part of the pyReef carbonate platform modelling application.     !!
!!                                                                                   !!
!!  For full license and copyright information, please refer to the LICENSE.md file  !!
!!  located at the project root, or contact the authors.                             !!
!!                                                                                   !!
!!~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~!!

! Main wave modelling data class.

module classdata

  ! SP grid model values
  integer::sp_n,sp_m
  integer::xlen,ylen,xclen,yclen,stratal_x,stratal_y
  real::stratal_xo,stratal_yo,stratal_dx,stratal_xm,stratal_ym
  real::sea_level
  real,dimension(:,:),allocatable::sp_topo

  ! Forecast parameters
  real::wave_base
  real,dimension(8)::forecast_param

  ! Number of hindcast scenarios
  type hindcast_param
    ! Significant wave height (in metres).
    real::hs
    ! Wave period of the energy spectrum
    real::per
    ! Peak wave direction.
    real::dir
    ! Coefficient of directional spreading.
    real::dd
    ! Wind velocity at 10 m elevation (m/s).
    real::wvel
    ! Wind direction.
    real::wdir
  end type hindcast_param
  type(hindcast_param)::hindcast

end module classdata
! ============================================================================
module mpidata

#include "mpif.h"

  ! Error code
  integer::ierr
  ! Processor ID
  integer::iam
  ! Number of processors
  integer::nprocs

end module mpidata
! ============================================================================
module miscdata

  ! Input / output files
  character(len=128)::finput
  character(len=128)::xyzfile
  character(len=128)::swaninput
  character(len=128)::swaninfo
  character(len=128)::swanbot
  character(len=128)::swanout
  character(len=128)::outdir
  character(len=128)::outdir1
  character(len=128)::h5data

  ! MPI integer type communicator
  integer::int_type
  ! MPI double type communicator
  integer::dbl_type
  ! MPI double type communicator
  integer::real_type
  ! MPI logical type communicator
  integer::lgc_type
  ! MPI max type communicator
  integer::max_type
  ! MPI min type communicator
  integer::min_type
  ! MPI sum type communicator
  integer::sum_type
  ! SPModel communicator
  integer::ocean_comm_world

contains

  ! ============================================================================
  subroutine term_command(cmds)

    logical(4)::result
    character(len=128)::cmds
    result=.false.
    ! INTEL FORTRAN COMPILER
    !result=systemqq(cmds)
    ! GNU FORTRAN COMPILER
    call system(cmds)

    return

  end subroutine term_command
  ! ============================================================================
  subroutine append_str2(stg1,stg2)

    integer::l1,l2
    character(len=128)::stg1,stg2

    l1=len_trim(stg1)
    l2=len_trim(stg2)
    stg1(l1+1:l1+l2)=stg2

    return

  end subroutine append_str2
  ! ============================================================================
  subroutine append_str(stg1,stg2)

    integer::l1,l2
    character(len=128)::stg1,stg2

    l1=len_trim(stg1)
    l2=len_trim(stg2)
    stg1(l1+1:l1+l2)=stg2
    call noblnk(stg1)

    return

  end subroutine append_str
  ! ============================================================================
  subroutine noblnk(string)

    integer::i,j,lg
    character(len=128)::string

    lg=len(string)
    do
      if(lg<=0.or.string(lg:lg)/=' ') exit
      lg=lg-1
    enddo
    if(lg>0)then
      ! find first non-blank character
      i=1
      do
        if(i>lg.or.(string(i:i)/=' '.and.string /= ' ')) exit
        i=i+1
      enddo
      ! determine end of continuous (non-blank) string
      j=i
      do
        if(j>lg)then
          exit
        elseif(string(j:j)==' ')then
          exit
        elseif(string=='  ')then
          exit
        elseif(j==128)then
          exit
        endif
        j=j+1
      enddo
      ! j points to first blank position or past end of string; adjust to last
      ! non-blank position in string
      j=min(j-1,lg)
      string=string(i:j)
      if(j<len(string)) string(j+1:len(string))=' '
    else
       ! there were only blanks in string
       string=' '
    endif

    return

  end subroutine noblnk
  ! ============================================================================
  subroutine addpath(fname)

    integer:: pathlen,flen
    character(len=128)::fname,dname,dummy

    ! for files to be read,they'll be in the session path
    dname=' '
    call noblnk(outdir)
    pathlen=len_trim(outdir)
    dname(1:pathlen)=outdir
    dname(pathlen+1:pathlen+1)='/'
    pathlen=pathlen+1
    call noblnk(fname)
    flen=len_trim(fname)
    dummy=' '
    dummy=fname
    fname=' '
    fname(1:pathlen)=dname(1:pathlen)
    fname(pathlen+1:pathlen+flen)=dummy(1:flen)

    return

  end subroutine addpath
  ! ============================================================================
  subroutine addpath1(fname)

    integer:: pathlen,flen
    character(len=128)::fname,dname,dummy

    ! for files to be read,they'll be in the session path
    dname=' '
    call noblnk(outdir1)
    pathlen=len_trim(outdir1)
    dname(1:pathlen)=outdir1
    dname(pathlen+1:pathlen+1)='/'
    pathlen=pathlen+1
    call noblnk(fname)
    flen=len_trim(fname)
    dummy=' '
    dummy=fname
    fname=' '
    fname(1:pathlen)=dname(1:pathlen)
    fname(pathlen+1:pathlen+flen)=dummy(1:flen)

    return

  end subroutine addpath1
  ! ============================================================================

end module miscdata
! ============================================================================
module wavegrid

  use mpidata
  use classdata
  use miscdata

  implicit none

contains

  ! =====================================================================================
  subroutine create_swan_data

    logical::found
    integer::i,j,l1,scn
    character(len=128)::command,stg
    real::x,y,z,pi

    ! SPM grid extent
    stratal_x=sp_n
    stratal_y=sp_m

    ! Get maximum bathymetry value
    if(allocated(sp_topo)) deallocate(sp_topo)
    allocate(sp_topo(sp_m,sp_n))
    open(unit=17,file=trim(xyzfile))
    rewind(17)
    do i=1,sp_m
      do j=1,sp_n
        read(17,*)l1,x,y,z
        sp_topo(i,j)=z+50-sea_level
      enddo
    enddo
    close(17)

    ! Allocate forecasts
    pi=4.*atan(1.)
    forecast_param(1)=hindcast%hs
    forecast_param(2)=hindcast%per
    forecast_param(3)=hindcast%dir
    forecast_param(4)=hindcast%dd
    forecast_param(5)=hindcast%wvel
    forecast_param(6)=hindcast%wdir
    forecast_param(7)=forecast_param(5)*sin(forecast_param(6)*180./pi)
    forecast_param(8)=forecast_param(5)*cos(forecast_param(6)* 180./pi)

    ! Create the output directory
    outdir1=''
    if(iam==0)then
        command=' '
        command(1:6)='rm -r '
        l1=len_trim(outdir)
        command(7:l1+7)=outdir
        call term_command(command)
        command=' '
        command(1:6)='mkdir '
        l1=len_trim(outdir)
        command(7:l1+7)=outdir
        call term_command(command)
        command(l1+7:l1+7)='/'
        stg=''
        stg(1:l1+7)=command(1:l1+7)
        stg(l1+8:l1+13)='swan'
        call term_command(stg)
        outdir1(1:l1)=outdir(1:l1)
        outdir1(l1+1:l1+7)='/swan'
        call noblnk(outdir1)
    endif
    call mpi_bcast(outdir,128,mpi_character,0,ocean_comm_world,ierr)
    call mpi_bcast(outdir1,128,mpi_character,0,ocean_comm_world,ierr)

    return

  end subroutine create_swan_data
  ! =====================================================================================

end module wavegrid
