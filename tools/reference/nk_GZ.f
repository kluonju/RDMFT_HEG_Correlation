ccccc momentum distribution of 3D unpolarized uniform electron gas
ccccc valid for rs < 12
ccccc from Gori-Giorgi and Ziesche, PRB 66, 235116 (2002)
      subroutine nofk(r,rrs,ank)
ccc r -> k/kF
ccc rrs -> density parameter
ccc ank -> n(k/kF,rs)
      implicit double precision (a-h,o-z)
      real*8 n0
      common/dens/rs
      rs=rrs

      pi=dacos(-1.d0)
      alph=(4.d0/9.d0/pi)**(1.d0/3.d0)

      n0=an0(rs)
      aa=apar(rs)
      bb=bpar(rs)
      call parn1(rs,an1m,an1p)

      G0=3.353337d0

      fac=sqrt(4.d0*alph*rs/pi)
      fac2=alph*rs/2.d0/pi**2
      F02=8.984373d0
      fac4=pi*(1.d0-log(2.d0))/6.d0/F02*abs((n0-an1m))/G0
      fac4=sqrt(fac4)

      if(r.le.1.d0) then

         xk=aa*fac2*G0/(n0-an1m)*(1.d0-r)/fac
         xk=xk+bb/r/fac2/2.d0*fac4*(1.d0-r)**2
         call kulik(xk,Gk)
         ank=n0-(n0-an1m)/G0*Gk
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      else
         fac3=sqrt(3.d0*pi*(1.d0-log(2.d0))*an1p/G0/g0GP(rs))

         xk=aa*fac2*G0/an1p*(r-1.d0)/fac
         xk=xk+fac3*fac**2*((r-1.d0)/fac)**4
         call kulik(xk,Gk)
         ank=an1p/G0*Gk
      endif
      return
      end


      subroutine kulik(x1,Gk)

      implicit double precision (a-h,o-z)
      parameter (nndim=1500)
      dimension s1(nndim),y1(nndim),ui(nndim),wi(nndim)
      common/vars/xx
      
      xx=x1
 
      if(x1.le.1.4d-3) then
         G0=3.353337
         pi=dacos(-1.d0)
         cc=pi*(pi/4.d0+sqrt(3.d0))
         Gk=G0+cc*x1*log(x1)
      endif

      ndm=900
      
      rhomin = -6.3d0
      h      = .0218d0
      alpha  = 0.d0
c      do 115 ndm=nmin,nmax
         r = exp(rhomin)
         do 110 i = 1, ndm
            rhx = rhomin + (i-1)*h
            do 111 k=1,10
               rhop = log(r) + alpha*r
               r = r + (rhx-rhop)*r/(1.d0+alpha*r)
 111        continue
            ui(i) = r
            s1(i) = xkul(r)
            wi(i) = r/(1.d0+alpha*r)
 110     continue
         call xint(s1,y1,0.d0,wi,h,ndm)
         Gk=y1(ndm)

      return
      end

      double precision function xkul(u)
      implicit double precision(a-h,o-z)
      common/vars/xx
      x=xx
      at=atan(1.d0/u)
      Ru=1.d0-u*at
      Rud=-at+u/(u*u+1.d0)
      y=x/sqrt(Ru)
      at2=atan(1.d0/y)
      xkul=-Rud/(Ru-x*x/u/u)*(at-y/u*at2)
      return
      end


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc INGREDIENTS cccccccccccccccccccccccccccccccccccccccccccccccccccc


cccc on top pair density

      double precision function g0GP(x)
ccc g(0) from Gori-Giorgi and Perdew, PRB 64, 155102 (2000) cccc
      implicit double precision(a-h,o-z)
      c=0.08183d0
      d=-0.01277d0
      e=0.001859d0
      dex=0.7524d0
      b=0.7317d0-dex
      g0GP=(1.d0-b*x+c*x**2+d*x**3+e*x**4)*exp(-dex*x)
      g0GP=g0GP/2.d0
      return
      end

ccc curvature near k = 0
ccc interpolation between H and L density limits
      double precision function bpar(x)
      implicit double precision (a-h,o-z)
      pi=dacos(-1.d0)
      alpha=(9.d0*pi/4.d0)**(-1.d0/3.d0)
      F2=8.984373d0
      gWC=2.65d0/2.d0
      bhd=(alpha/pi**2)**2*F2
      bld=4.d0/(3.d0*sqrt(pi))/(2.d0/3.d0*gWC*alpha**2)**(5.d0/2.d0)
      bpar=sqrt((1.d0+bhd*x**(13.d0/4.d0)/bld))
      return
      end


ccc coeff. of  (1-k)log(1-k)
ccc Pade' of result from sum rules
      double precision function apar(x)
      implicit double precision (a-h,o-z)
ccc from sum rules
      gg1             = -78.8682d0      
      gg2             = -0.0989941d0     
      gg3             = -68.5997d0       
      gg4             = -17.6829d0       
      gg5             = 38.1159d0        
      gg6             = -0.114831d0
      apar=(1.d0+gg1*x**(1.d0/4.d0)+gg2*sqrt(x))/(1.d0+
     $     gg3*x**(1.d0/4.d0)+gg4*x+gg5*x**(1.d0/2.d0)+
     $     gg2*abs(gg6)*x**4)

      return
      end

cccc n(k=0,rs)
cccc Pade' of TY data
      double precision function an0(x)
      implicit double precision (a-h,o-z)
      real*8 n
      pi=dacos(-1.d0)
      cf=(9.d0*pi/4.d0)**(1.d0/3.d0)
      chd=4.11234d0/cf**2/pi**4
cccc link between H and L density limit + Pade' of TY
      n=2.5d0
      gWC=2.65d0/2.d0
      dwc=9.d0*sqrt(6.d0)*sqrt(pi)/4.d0/gWC**1.5d0
      t1= 0.0586359d0
      t2= 0.0336518d0 

      an0=(1.d0+t1**2*x**2+dwc*t2**2*x**n)/(1.d0+(t1**2+chd)*x**2
     $     +t2**2*x**(n+3.d0/4.d0))

      return
      end

      subroutine parn1(x,an1m,an1p)
      implicit double precision (a-h,o-z)
      c1=0.177038d0/2.d0
      win=3.8847405713855d0
ccc n1- from sum rules
      v1= -0.0679793d0  
      v2= -0.00102846d0 
      v3= 0.000189111d0 
      v4= -0.0086838d0  
      v5= 6.87109d-05   
      an1m=(1.d0+v1*x+v2*x**2+v3*x**3)/(1.d0+(v1+c1)*x+
     $     v4*x**2+v5*x**3+v3/win*x**(3+3.d0/4.d0))
ccc model for n1+ with interpol. between H and L D
      q1=0.45d0
      an1p=c1*x/(1.d0+q1*x**(1.d0/2.d0)+c1/win*x**(7.d0/4.d0))
      
      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c kinetic energy of correlation: in HARTREE!!!
c Perdew & Wang PRB 45, 13244 (1992) + virial theorem
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine GPW(x,Ac,alfa1,beta1,beta2,beta3,beta4,G,Gd)
ccc Gd is d/drs G
      implicit real*8(a-h,o-z)
      G=-2.d0*Ac*(1.d0+alfa1*x)*dlog(1.d0+1.d0/(2.d0*
     $     Ac*(beta1*x**0.5d0+
     $     beta2*x+beta3*x**1.5d0+beta4*x**2)))
      Gd=(1.d0+alfa1*x)*(beta2+beta1/(2.d0*sqrt(x))+3.d0*beta3*
     $     sqrt(x)/2.d0+2.d0*beta4*x)/((beta1*sqrt(x)+beta2*x+
     $     beta3*x**(3.d0/2.d0)+beta4*x**2)**2*(1.d0+1.d0/
     $     (2.d0*Ac*(beta1*sqrt(x)+beta2*x+beta3*x**(3.d0/2.d0)+
     $     beta4*x**2))))-2.d0*Ac*alfa1*dlog(1.d0+1.d0/(2.d0*Ac*
     $     (beta1*sqrt(x)+beta2*x+beta3*x**(3.d0/2.d0)+
     $     beta4*x**2)))
      return
      end

      subroutine tcPW(x,y,tc)
c in Hartree; ec=ec(rs,zeta)
c x -> rs; y -> zeta
ccc ecd is d/drs ec
      implicit real*8(a-h,o-z)
      pi=dacos(-1.d0)
      
      f02=4.d0/(9.d0*(2.d0**(1.d0/3.d0)-1.d0))

      ff=((1.d0+y)**(4.d0/3.d0)+(1.d0-y)**(4.d0/3.d0)-
     $     2.d0)/(2.d0**(4.d0/3.d0)-2.d0)

      aaa=(1.d0-dlog(2.d0))/pi**2
      call  GPW(x,aaa,0.21370d0,7.5957d0,3.5876d0,
     $     1.6382d0,0.49294d0,G,Gd)
      ec0=G
      ec0d=Gd

      aaa=aaa/2.d0
      call GPW(x,aaa,0.20548d0,14.1189d0,6.1977d0,
     $     3.3662d0,0.62517d0,G,Gd)
      ec1=G
      ec1d=Gd
      call GPW(x,0.016887d0,0.11125d0,10.357d0,3.6231d0,
     $     0.88026d0,0.49671d0,G,Gd)
      alfac=-G
      alfacd=-Gd

      ec=ec0+alfac*ff/f02*(1.d0-y**4)+(ec1-ec0)*ff*y**4
      ecd=ec0d+alfacd*ff/f02*(1.d0-y**4)+(ec1d-ec0d)*
     $     ff*y**4

      tc=-ec-x*ecd

      return
      end


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c INTEGRATION SUBROUTINE
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine xint( x, y, y1, wr, h, n )
      implicit real*8(a-h,o-z)
      parameter (mint=2000)
      dimension x(mint), y(mint),wr(mint)
c***********************************************************************
cengel
c  xint calculates the indefinite integral
c
c   y(i) = int_0^ar(i) dr x(r)
c
c  for all i <= n. y1 is the correction term to be provided in the
c  call which represents the integral from 0 to the first mesh point
c  ar(1). xint is based on 6-point formulas.
c
c  note that this subroutine is vectorizable. however, vectorization
c  requires more than the minimum number of additions which could
c  affect the numerical accuracy.
cengel
c***********************************************************************
      ca=wr(1)*x(1)
      cb=wr(2)*x(2)
      cc=wr(3)*x(3)
      cd=wr(4)*x(4)
      ce=wr(5)*x(5)
      cf=wr(6)*x(6)
      wa=   h *  11.d0/1440.d0
      wb= - h *  93.d0/1440.d0
      wc=   h * 802.d0/1440.d0
      y(1)=y1
      y(2)=y(1)+h*(   475.d0*ca + 1427.d0*cb -  798.d0*cc
     &             +  482.d0*cd -  173.d0*ce +   27.d0*cf)/1440.d0
      y(3)=y(2)+h*(-   27.d0*ca +  637.d0*cb + 1022.d0*cc
     &             -  258.d0*cd +   77.d0*ce -   11.d0*cf)/1440.d0
      do 2100 i=4,n-2
         cf = wr(i+2)*x(i+2)
         y(i) = y(i-1)+wa*(ca+cf)+wb*(cb+ce)+wc*(cc+cd)
         ca = cb
         cb = cc
         cc = cd
         cd = ce
         ce = cf
 2100 continue
      y(n-1)=y(n-2)+h*(- 11.d0*wr(n-5)*x(n-5) + 77.d0*ca - 258.d0*cb
     &                 + 1022.d0*cc + 637.d0*cd - 27.d0*ce )/1440.d0
      y(n)=y(n-1)+h*(  27.d0*wr(n-5)*x(n-5) - 173.d0*ca + 482.d0*cb
     &               - 798.d0*cc + 1427.d0*cd + 475.d0*ce )/1440.d0
      return
      end
c


