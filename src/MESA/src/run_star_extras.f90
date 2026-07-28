! ***********************************************************************
!
!   Copyright (C) 2010-2019  Bill Paxton & The MESA Team
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful, 
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************

module run_star_extras

   use star_lib
   use star_def
   use const_def
   use math_lib
   use rates_def
   use chem_def

   implicit none

   ! s% xtra

   ! s% lxtra

   ! Extra controls options

   ! s% x_ctrl
   integer, parameter :: i_turb_constant = 1
   integer, parameter :: i_turb_exponent = 2
   integer, parameter :: i_turb_reference = 3

   ! s% x_integer_ctrl


   ! s% x_logical_ctrl


   ! For saving routine
   logical :: first_step = .true.
   real(dp), dimension(9), parameter :: tgt_center_h1 = [0.7d0, 0.6d0, 0.5d0, 0.4d0, 0.3d0, 0.2d0, 0.1d0, 0.05d0, 0.01d0]
   logical, dimension(SIZE(tgt_center_h1)) :: saved = .false.
   integer :: i_tgt_center_h1 = 1
   real(dp), parameter :: eps_Xc = 1d-5

   ! these routines are called by the standard run_star check_model
contains

   subroutine extras_controls(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      ! this is the place to set any procedure pointers you want to change
      ! e.g., other_wind, other_mixing, other_energy  (see star_data.inc)


      ! the extras functions in this file will not be called
      ! unless you set their function pointers as done below.
      ! otherwise we use a null_ version which does nothing (except warn).

      s% extras_startup => extras_startup
      s% extras_start_step => extras_start_step
      s% extras_check_model => extras_check_model
      s% extras_finish_step => extras_finish_step
      s% extras_after_evolve => extras_after_evolve
      s% how_many_extra_history_columns => how_many_extra_history_columns
      s% data_for_extra_history_columns => data_for_extra_history_columns
      s% how_many_extra_profile_columns => how_many_extra_profile_columns
      s% data_for_extra_profile_columns => data_for_extra_profile_columns

      s% how_many_extra_history_header_items => how_many_extra_history_header_items
      s% data_for_extra_history_header_items => data_for_extra_history_header_items
      s% how_many_extra_profile_header_items => how_many_extra_profile_header_items
      s% data_for_extra_profile_header_items => data_for_extra_profile_header_items

      s% other_D_mix => turbulent_mixing
      s% other_adjust_mlt_gradT_fraction => other_adjust_mlt_gradT_fraction_Peclet

   end subroutine extras_controls


   subroutine extras_startup(id, restart, ierr)
      integer, intent(in) :: id
      logical, intent(in) :: restart
      integer, intent(out) :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end subroutine extras_startup


   integer function extras_start_step(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      integer :: i
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      extras_start_step = 0

      ! Check which center_h1 is currrent target
      do i = 1, 8
         if ((s% center_h1 <= tgt_center_h1(i)) .and. (s% center_h1 > tgt_center_h1(i + 1))) then
            i_tgt_center_h1 = i + 1
            exit
         end if
      end do

   end function extras_start_step


   ! returns either keep_going, retry, or terminate.
   integer function extras_check_model(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      integer :: i

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      extras_check_model = keep_going

      ! if you want to check multiple conditions, it can be useful
      ! to set a different termination code depending on which
      ! condition was triggered.  MESA provides 9 customizeable
      ! termination codes, named t_xtra1 .. t_xtra9.  You can
      ! customize the messages that will be printed upon exit by
      ! setting the corresponding termination_code_str value.
      ! termination_code_str(t_xtra1) = 'my termination condition'

      ! by default, indicate where (in the code) MESA terminated

      ! Retry if overshot target Xc and have not saved a profile
      if (.not. saved(i_tgt_center_h1)) then
         if (s% center_h1 < tgt_center_h1(i_tgt_center_h1) - eps_Xc) then
            extras_check_model = retry
         end if
      end if

      if (extras_check_model == terminate) s% termination_code = t_extras_check_model
   end function extras_check_model


   ! returns either keep_going or terminate.
   ! note: cannot request retry; extras_check_model can do that.
   integer function extras_finish_step(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      logical :: save_now
      real(dp) :: max_dXc, max_dXc_hard
      integer :: i

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      extras_finish_step = keep_going

      ! to save a profile,
      ! s% need_to_save_profiles_now = .true.
      ! to update the star log,
      ! s% need_to_update_history_now = .true.

      ! extras_check_model will ensure this
      save_now = (abs(s% center_h1 - tgt_center_h1(i_tgt_center_h1)) < eps_Xc)  ! At target Xc
      save_now = (save_now .and. (.not. saved(i_tgt_center_h1)))  ! Only save one profile per target Xc
      if (save_now) then
         s% need_to_save_profiles_now = .true.
         saved(i_tgt_center_h1) = .true.
      end if

      write(*,*) tgt_center_h1(i_tgt_center_h1), s% center_h1

      if (saved(SIZE(saved))) then
         extras_finish_step = terminate
      end if

      ! see extras_check_model for information about custom termination codes
      ! by default, indicate where (in the code) MESA terminated
      if (extras_finish_step == terminate) s% termination_code = t_extras_finish_step
   end function extras_finish_step


   integer function how_many_extra_history_columns(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      how_many_extra_history_columns = 2

   end function how_many_extra_history_columns


   subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
      integer, intent(in) :: id, n
      character (len = maxlen_history_column_name) :: names(n)
      real(dp) :: vals(n)
      integer, intent(out) :: ierr
      type (star_info), pointer :: s

      integer :: k, k2, k3, max_eps_h_k
      character (len = 1) :: pm
      real(dp) :: nu_q
      real(dp) :: out(30, 2)
      integer :: out_int(2, 2)

      integer :: i, k_l
      real(dp) :: r_bCZ, m_bCZ, alfa, beta

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      k = 1
      ! Calculate bottom of conv envelope
      max_eps_h_k = maxloc(s% eps_nuc_categories(ipp, 1:s% nz) + s% eps_nuc_categories(icno, 1:s% nz), 1)
      r_bCZ = -1d99
      m_bCZ = -1d99
      do i = max_eps_h_k, 2, -1
         if ((s% gradr(i) < s% grada(i)) .and. (s% gradr(i - 1) > s% grada(i - 1)) .and. (s% xa(s% net_iso(ih1),i) > 0.5)) then  ! botCZ between i and i-1
            alfa = (s% gradr(i) - s% grada(i)) / ((s% gradr(i) - s% grada(i)) - (s% gradr(i - 1) - s% grada(i - 1)))
            beta = 1 - alfa
            r_bCZ = (beta * s% r(i)**3 + alfa * s% r(i - 1)**3)**(1d0/3d0) / rsun
            m_bCZ = (beta * s% m(i) + alfa * s% m(i - 1)) / msun
            exit
         end if
      end do
      names(k) = 'r_botCZ'
      vals(k) = r_bCZ
      k = k + 1
      names(k) = 'm_botCZ'
      vals(k) = m_bCZ
      k = k + 1

   end subroutine data_for_extra_history_columns


   integer function how_many_extra_profile_columns(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      how_many_extra_profile_columns = 0
   end function how_many_extra_profile_columns


   subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
      integer, intent(in) :: id, n, nz
      character (len = maxlen_profile_column_name) :: names(n)
      real(dp) :: vals(nz, n)
      integer, intent(out) :: ierr
      type (star_info), pointer :: s
      integer :: k
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      ! note: do NOT add the extra names to profile_columns.list
      ! the profile_columns.list is only for the built-in profile column options.
      ! it must not include the new column names you are adding here.

   end subroutine data_for_extra_profile_columns


   integer function how_many_extra_history_header_items(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      how_many_extra_history_header_items = 0
   end function how_many_extra_history_header_items


   subroutine data_for_extra_history_header_items(id, n, names, vals, ierr)
      integer, intent(in) :: id, n
      character (len = maxlen_history_column_name) :: names(n)
      real(dp) :: vals(n)
      type(star_info), pointer :: s
      integer, intent(out) :: ierr
      ierr = 0
      call star_ptr(id, s, ierr)
      if(ierr/=0) return

      ! here is an example for adding an extra history header item
      ! also set how_many_extra_history_header_items
      ! names(1) = 'mixing_length_alpha'
      ! vals(1) = s% mixing_length_alpha

   end subroutine data_for_extra_history_header_items


   integer function how_many_extra_profile_header_items(id)
      integer, intent(in) :: id
      integer :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      how_many_extra_profile_header_items = 0
   end function how_many_extra_profile_header_items


   subroutine data_for_extra_profile_header_items(id, n, names, vals, ierr)
      integer, intent(in) :: id, n
      character (len = maxlen_profile_column_name) :: names(n)
      real(dp) :: vals(n)
      type(star_info), pointer :: s
      integer, intent(out) :: ierr
      ierr = 0
      call star_ptr(id, s, ierr)
      if(ierr/=0) return

      ! here is an example for adding an extra profile header item
      ! also set how_many_extra_profile_header_items
      ! names(1) = 'mixing_length_alpha'
      ! vals(1) = s% mixing_length_alpha

   end subroutine data_for_extra_profile_header_items

   subroutine extras_after_evolve(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type (star_info), pointer :: s
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end subroutine extras_after_evolve

   subroutine turbulent_mixing(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type (star_info), pointer :: s

      real(dp) :: f_turb, n_turb, T_turb, new_Dmix, DHe_0, Rho_0
      real(dp) :: alfa, beta
      integer :: k

      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      f_turb = s% x_ctrl(i_turb_constant)
      n_turb = s% x_ctrl(i_turb_exponent)
      T_turb = s% x_ctrl(i_turb_reference)

      do k = 1, s% nz
         if (s% lnT(k) > log(T_turb)) then
            exit
         end if
      end do
      alfa = (s% lnT(k) - log(T_turb)) / (s% lnT(k) - s% lnT(k - 1))
      beta = 1d0 - alfa

      Rho_0 = (beta * s% Rho(k) + alfa * s% Rho(k - 1))
      DHe_0 = (3.3d-15 * T_turb**2.5d0) / (4d0 * Rho_0 * log(1 + 1.125d-16 * T_turb**3 * Rho_0))

      do k = 1, s% nz
         new_Dmix = f_turb * DHe_0 * (s% Rho(k)/Rho_0)**n_turb
         new_Dmix = min(new_Dmix, 1d10)  ! Upper limit for numerical stability if using relax_tau_factor
         if ((new_Dmix > s% D_mix(k)) .and. (new_Dmix > 1d-3)) then
            s% D_mix(k) = new_Dmix
            s% mixing_type(k) = minimum_mixing
         end if
      end do

   end subroutine turbulent_mixing


   ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   ! Michielsen+ 2023
   ! Adjust temperature gradient in overshoot zone to be adiabatic (Pe>1d2) or radiative (Pe<1d-2) based upon the Peclet number,
   ! with a gradual transition between the two regimes.
   ! Only works if conv_premix = .true. since the last iteration in a time step has NaNs in D_mix if conv_premix = .false.
   ! The other hook on the next line needs to be included in run_star_extras to use this routine.
   ! s% other_adjust_mlt_gradT_fraction => other_adjust_mlt_gradT_fraction_Peclet
   subroutine other_adjust_mlt_gradT_fraction_Peclet(id, ierr)
   integer, intent(in) :: id
   integer, intent(out) :: ierr
   type(star_info), pointer :: s
   real(dp) :: fraction, Peclet_number, conductivity, Hp       ! f is fraction to compose grad_T = f*grad_ad + (1-f)*grad_rad
   integer :: k
   logical, parameter :: DEBUG = .FALSE.

   ierr = 0
   call star_ptr(id, s, ierr)
   if (ierr /= 0) return

   if (s%D_mix(1) .ne. s%D_mix(1)) return  ! To ignore iterations where Dmix and gradT are NaNs

   if (s%num_conv_boundaries .lt. 1) then  ! Is zero at initialisation of the run
   if (DEBUG) then
      write(*,*) ' skip since there are no convective boundaries'
   end if
   return
   endif

   do k= s%nz, 1, -1
      if (s%D_mix(k) <= s% min_D_mix) exit

      conductivity = 16.0_dp * boltz_sigma * pow3(s% T(k)) / ( 3.0_dp * s% opacity(k) * pow2(s% rho(k)) * s% cp(k) )
      Hp = s% Peos(k)/(s% rho(k)*s% grav(k)) ! Pressure scale height
      Peclet_number = s% conv_vel(k) * Hp * s% mixing_length_alpha / conductivity

      if (Peclet_number >= 100.0_dp) then
          fraction = 1.0_dp
      else if (Peclet_number .le. 0.01_dp) then
          fraction = 0.0_dp
      else
          fraction = (safe_log10(Peclet_number)+2.0_dp)/4.0_dp
      end if

      s% adjust_mlt_gradT_fraction(k) = fraction
   end do

   end subroutine other_adjust_mlt_gradT_fraction_Peclet

end module run_star_extras
