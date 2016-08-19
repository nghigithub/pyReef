!!~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~!!
!!                                                                                   !!
!!  This file forms part of the pyReef carbonate platform modelling application.     !!
!!                                                                                   !!
!!  For full license and copyright information, please refer to the LICENSE.md file  !!
!!  located at the project root, or contact the authors.                             !!
!!                                                                                   !!
!!~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~!!

! Main entry to SWAN wave modelling code.

module simswan

  use mpidata
  use wavegrid
  use SimWaves
  use classdata
  use miscdata

  implicit none

contains

  subroutine run(comm)

      real(kind=8)::time_st,time_ed
      integer :: comm

      ! start up MPI
      call mpi_comm_size(comm,nprocs,ierr)
      call mpi_comm_rank(comm,iam,ierr)
      ocean_comm_world=comm

      int_type=mpi_integer
      real_type=mpi_real
      dbl_type=mpi_double_precision
      lgc_type=mpi_logical
      max_type=mpi_max
      min_type=mpi_min
      sum_type=mpi_sum

      time_st=mpi_wtime( )

      !--
      ! Required data
      xyzfile='data/gab_topo.nodes'
      sp_n = 250
      sp_m = 249
      stratal_dx = 3000
      wave_base = 100.0
      !--
      hindcast%wvel=30.0
      hindcast%wdir=140.0
      !--
      outdir='GABcirc'
      !--

      ! Initialisation step for several calls
      ! Initialise swan model dataset and directory
      call create_swan_data
      call build_swan_model

      ! Compute the wave fields for each conditions:
      ! needs to update forecast based on next forecast_param for next run
      ! needs to update topo sp_topo sea_level
      call run_waves

      time_ed=mpi_wtime( )
      if(iam==0) print*,'Time elapse:',time_ed-time_st


      ! Finalisation
      ! Finalisation
      call wave_final

      return

  end subroutine run

end module simswan
