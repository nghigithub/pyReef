!!~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~!!
!!                                                                                   !!
!!  This file forms part of the pyReef carbonate platform modelling application.     !!
!!                                                                                   !!
!!  For full license and copyright information, please refer to the LICENSE.md file  !!
!!  located at the project root, or contact the authors.                             !!
!!                                                                                   !!
!!~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~!!

! Module for multi-lithology non-linear diffusion model based on landiff
! (https://github.com/tristan-salles/landiff).

module diffuse

    integer::lithNb
    real::dx2,topLay,dt

    real,dimension(:),allocatable::poroCoeff,diffCoeff

contains

    subroutine init(pyDiffH, pyPorosity, pyDiffusion, pyDx, pyDt, pyFacies)

      integer,intent(in) :: pyFacies
      real,intent(in) :: pyDt
      real,intent(in) :: pyDiffH
      real,dimension(pyFacies),intent(in) :: pyPorosity
      real,dimension(pyFacies),intent(in) :: pyDiffusion
      real,intent(in) :: pyDx

      lithNb = pyFacies
      topLay = pydiffH
      dx2 = pyDx*pyDx
      dt = pyDt

      if(allocated(poroCoeff)) deallocate(poroCoeff)
      if(allocated(diffCoeff)) deallocate(diffCoeff)
      allocate(poroCoeff(lithNb))
      allocate(diffCoeff(lithNb))

      poroCoeff = pyPorosity
      diffCoeff = pyDiffusion

    end subroutine init

    subroutine run(pySed, pyElev, outZ, outSed, pyNx, pyNy, pyFNb)

      integer,intent(in) :: pyNx
      integer,intent(in) :: pyNy
      integer,intent(in) :: pyFNb
      real,dimension(pyNx,pyNy),intent(in) :: pyElev
      real,dimension(pyNx,pyNy,pyFNb),intent(in) :: pySed

      real,dimension(pyNx,pyNy),intent(out) :: outZ
      real,dimension(pyNx,pyNy,pyFNb),intent(out)::outSed

      integer::i,j,k

      real::sum,ax,ay,diff
      real::halfxp,halfxm,halfyp,halfym

      real,dimension(pyNx,pyNy,pyFNb)::tmpLitho

      ! First we solve the equation on elevation
      do j=2,pyNy-1
        do i=2,pyNx-1
          sum=0.
          do k=1,lithNb
            diff=diffCoeff(k)
            halfxp=0.5*diff*(pySed(i,j,k)+pySed(i+1,j,k))
            halfxm=0.5*diff*(pySed(i,j,k)+pySed(i-1,j,k))
            halfyp=0.5*diff*(pySed(i,j,k)+pySed(i,j+1,k))
            halfym=0.5*diff*(pySed(i,j,k)+pySed(i,j-1,k))
            ax=halfxp*(pyElev(i+1,j)-pyElev(i,j))-halfxm*(pyElev(i,j)-pyElev(i-1,j))
            ay=halfyp*(pyElev(i,j+1)-pyElev(i,j))-halfym*(pyElev(i,j)-pyElev(i,j-1))
            sum=sum+ax/dx2+ay/dx2
          enddo
          outZ(i,j)=pyElev(i,j)+sum*dt
        enddo
      enddo

      ! Ghost cells
      outZ(1,1)=pyElev(1,1)
      outZ(1,pyNy)=pyElev(1,pyNy)
      outZ(pyNx,1)=pyElev(pyNx,1)
      outZ(pyNx,pyNy)=pyElev(pyNx,pyNy)
      outZ(1,2:pyNy-1)=pyElev(1,2:pyNy-1)
      outZ(pyNx,2:pyNy-1)=pyElev(pyNx,2:pyNy-1)
      outZ(2:pyNx-1,1)=pyElev(2:pyNx-1,1)
      outZ(2:pyNx-1,pyNy)=pyElev(2:pyNx-1,pyNy)

      ! Solve sediment proportion equation
      tmpLitho=0.
      do k=1,lithNb-1
        do j=2,pyNy-1
          do i=2,pyNx-1
            diff=diffCoeff(k)
            halfxp=0.
            if(outZ(i-1,j)>outZ(i+1,j))then
              halfxp=dt*(diff*pySed(i,j,k)-diff*pySed(i-1,j,k))/(2.*dx2)
            else
              halfxp=dt*(diff*pySed(i+1,j,k)-diff*pySed(i,j,k))/(2.*dx2)
            endif
            halfyp=0.
            if(outZ(i,j-1)>outZ(i,j+1))then
              halfyp=dt*(diff*pySed(i,j,k)-diff*pySed(i,j-1,k))/(2.*dx2)
            else
              halfyp=dt*(diff*pySed(i,j+1,k)-diff*pySed(i,j,k))/(2.*dx2)
            endif
            sum=halfxp*(outZ(i+1,j)-outZ(i-1,j))+halfyp*(outZ(i,j+1)-outZ(i,j-1))
            tmpLitho(i,j,k)=(sum+pySed(i,j,k)*topLay)/(topLay+outZ(i,j)-pyElev(i,j))
          enddo
        enddo
      enddo

      ! Get all lithology proportions
      outSed = 0.
      do j=2,pyNy-1
        do i=2,pyNx-1
          sum=0.
          do k=1,lithNb-1
            sum=sum+tmpLitho(i,j,k)
            outSed(i,j,k)=tmpLitho(i,j,k)
          enddo
          if(sum>=1.)then
            diff=1./sum
            do k=1,lithNb-1
              outSed(i,j,k)=diff*outSed(i,j,k)
            enddo
            outSed(i,j,lithNb)=0.
          else
            outSed(i,j,lithNb)=1.-sum
          endif
        enddo
      enddo

    end subroutine run

end module diffuse
