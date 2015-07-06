! Angelica de Oliveira-Costa & Max Tegmark 2007
! f2py fixup Sean Passmoor 2015

	subroutine get_freq(nu,output,path)
	implicit  none
	integer ncomp, nside, npix
   	parameter(ncomp=3,nside=512,npix=12*512**2)
	integer ncomploaded, i, j, lnblnk
	real*8 nu, f(ncomp+1), norm, A(npix,ncomp), t,output(npix)
	character*180 infile!, outfile
	character*180 path
Cf2py   intent(out) output
Cf2py   intent(in) nu
Cf2py   intent(in) path
	!print *,'Frequency at which to make a map (in MHz)?'
	!read *,nu
	!print *,'Name of file in which to save the map?'
	!read *,outfile
	!print *,'Making sky map at frequeny__________',nu
	!print *,'Outfile_____________________________',outfile(1:lnblnk(outfile))
	
	call LoadComponents(ncomploaded,path)
	if (ncomploaded.ne.ncomp) stop 'DEATH ERROR: WRONG NUMBER OF COMPONENTS LOADED'
	call ComputeComponents(nu,ncomp,f)
	norm = f(ncomp+1)
	infile = TRIM(path)//'/component_maps_408locked.dat'
	!print *,'Loading ',infile(1:lnblnk(infile))
	open(2,file=infile,status='old')
	do i=1,npix
 	  read(2,*) (A(i,j),j=1,ncomp)
	end do	
	close(2)
	
	!print *,'Saving ',outfile(1:lnblnk(outfile))
	!open(3,file=outfile)	 	 
	do i=1,npix
	  t = 0
	  do j=1,ncomp
	    t = t + f(j)*A(i,j) 
	  end do
	  t = norm*t
	  output(i) = t
	  !write(3,*) t
	end do
	!close(3)
        return
 777    call usage
        end
	end

	integer function ntokens(line)
	character,intent(in):: line*(*)
	integer i, n, toks

	i = 1;
	n = len_trim(line)
	toks = 0
	ntokens = 0
	do while(i <= n)
	   do while(line(i:i) == ' ') 
	     i = i + 1
	     if (n < i) return
	   enddo
	   toks = toks + 1
	   ntokens = toks
	   do
	     i = i + 1
	     if (n < i) return
	     if (line(i:i) == ' ') exit
	   enddo
	enddo
	end function ntokens 
	
	subroutine LoadComponents(ncomp,path) ! ncomp = Number of components to load
	! Load the principal components from file and spline them for later use.
	! The "extra" component (ncomp+1) is the overall scaling - we spline its logarithm.
	implicit none
	integer  nmax, ncompmax, n, ncomp
	parameter(nmax=1000,ncompmax=11)
	real*8 x(nmax), y(nmax,ncompmax+1), ypp(nmax,ncompmax+1)
	common/PCA/x, y, ypp, n
	integer  i, lnblnk
	real*8   xn, scaling, tmp(nmax), yp0, yp1
	character*180 infile, comline
	character*180 path, s
	integer,external :: ntokens
	!
	infile = TRIM(path)//'/components.dat'
	! Count number of columns in the infile:'
	!comline = 'head -1 '//infile(1:lnblnk(infile))//' | wc | cut -c9-16 >qaz_cols.dat'
	!print *,'###'//comline(1:lnblnk(comline))//'###'
	!if (system(comline).ne.0) stop 'DEATH ERROR COUNTING COLUMNS'
	open (2,file=infile,status='old',err=777)
	read (2,'(a)',end=778,err=778) comline
	close(2)
	n = ntokens(comline)
	ncomp = n - 2
	!s = comline
	!n = count([len_trim(s) > 0,(s(i:i)/=' '.and.s(i:i)/=','.and.s(i+1:i+1)==' '.or.s(i+1:i+1)==',', i=1,len_trim(s)-1)])
	if (ncomp.lt.0       ) stop 'DEATH ERROR: TOO FEW  COMPONENTS.'
	if (ncomp.gt.ncompmax) stop 'DEATH ERROR: TOO MANY COMPONENTS.'
	n = 0
	open (2,file=infile,status='old')
555	read (2,*,end=666) xn, scaling, (tmp(i),i=1,ncomp)
	n = n + 1
	if (n.gt.nmax) pause 'n>nmax in LoadVector'
	x(n) = log(xn) ! We'll spline against lg(nu)
	do i=1,ncomp
	   y(n,i) = tmp(i)
	end do
	y(n,ncomp+1) = log(scaling)
	goto 555
666	close(2)
	!print *,ncomp,' components read from ',infile(1:lnblnk(infile)),' with',n,' spline points'
	yp0 = 1.d30 ! Imposes y''=0 at starting point
	yp1 = 1.d30 ! Imposes y''=0 at endpoint
	do i=1,ncomp+1
	   call myspline_r8(x,y(1,i),n,yp0,yp1,ypp(1,i))
	end do 
	return
777	stop 'DEATH ERROR 2 COUNTING COLUMNS'	
778	stop 'DEATH ERROR 3 COUNTING COLUMNS'	
	end 
	   
	subroutine ComputeComponents(nu,ncomp,a) ! Computes the principal components at frequency nu
	implicit none
	integer nmax, ncompmax, n, ncomp
	parameter(nmax=1000,ncompmax=11)
	real*8 x(nmax), y(nmax,ncompmax+1), ypp(nmax,ncompmax+1)
	common/PCA/x, y, ypp, n
	integer i
	real*8 a(ncompmax+1), nu, lnnu, scaling
	lnnu = log(nu)
	do i=1,ncomp+1
	  call mysplint_r8(x,y(1,i),ypp(1,i),n,lnnu,a(i))
	end do
	a(ncomp+1) = exp(a(ncomp+1)) ! The overall scaling factor
	return
	end
 	
      SUBROUTINE myspline_r8 (x,y,n,yp1,ypn,y2)
      ! From numerical recipes.
      INTEGER   n, NMAX, i, k
      REAL*8    yp1,ypn,x(n),y(n),y2(n)
      PARAMETER(NMAX=10000)	! Increased from 500 by Max
      REAL*8    p,qn,sig,un,u(NMAX)
      if (N.gt.NMAX)    pause 'SPLINE NMAX DEATH ERROR'          ! Added by Max
      if (x(1).gt.x(n)) pause 'SPLINE WARNING: x NOT INCREASING' ! Added by Max
      if (yp1.gt..99e30) then
        y2(1)=0.d0
        u (1)=0.d0
      else
        y2(1)=-0.5d0
        u (1)=(3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig  =(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p    = sig*y2(i-1)+2.d0
        y2(i)=(sig-1.d0)/p
        u (i)=(6.d0*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99e30) then
        qn=0.d0
        un=0.d0
      else
        qn=0.5d0
        un=(3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.d0)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END

      SUBROUTINE MYSPLINT_r8 (XA,YA,Y2A,N,X,Y)
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! From numerical recipes.
      ! Modified to be more robust when
      ! extrapolating - It is linear if X lies outside 
      ! the bounds.
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer  N, KLO, KHI, K
      REAL*8   XA(N),YA(N),Y2A(N), X, Y, H, A, B
      KLO=1
      KHI=N
1     IF (KHI-KLO.GT.1) THEN
        K=(KHI+KLO)/2
        IF(XA(K).GT.X)THEN
          KHI=K
        ELSE
          KLO=K
        ENDIF
      GOTO 1
      ENDIF
      H=XA(KHI)-XA(KLO)
      IF (H.EQ.0.d0) PAUSE 'Bad XA input.'
      if ((x-xa(1))*(x-xa(N)).gt.0.d0) then
        ! Outside bounds; do LINEAR extrapolation rather than cubic
        A = (YA(KHI)-YA(KLO))/H
        Y = ya(KLO) + A * (x-xa(KLO))
      else
        ! Within bounds; do cubic interpolation
        A=(XA(KHI)-X)/H
        B=(X-XA(KLO))/H
        Y=A*YA(KLO)+B*YA(KHI)+
     *      ((A**3-A)*Y2A(KLO)+(B**3-B)*Y2A(KHI))*(H**2)/6.
      end if
      RETURN
      END
