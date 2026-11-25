# Constraints for the TPU-style systolic array (chipflow reads this SDC subset; see chipflow/sdc.py).
#   python -m chipflow flow --design systolic --sdc designs/systolic_array.sdc --out runs/sa4_sdc

set period 2.0
create_clock -name core_clk -period $period [get_ports clk]
set_clock_uncertainty 0.05

# The neighbouring block launches operands from its own flip-flops and spends ~30% of the cycle
# getting them here; it drives each pin with a BUF_X4.
set data_in [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay [expr 0.3 * $period] -clock core_clk $data_in
set_driving_cell -lib_cell BUF_X4 $data_in

# The reader samples the accumulators ~30% of a cycle after they leave, into a 4 fF load.
set_output_delay [expr 0.3 * $period] -clock core_clk [all_outputs]
set_load 4.0 [all_outputs]

# The host reads acc only after the last operand has drained through the array (2N-1 cycles
# after the final input), so the accumulator pins get two cycles. -hold 1 keeps the hold check
# on the launch edge (without it, SDC moves hold one cycle later as well).
set_multicycle_path 2 -setup -to [get_ports acc]
set_multicycle_path 1 -hold  -to [get_ports acc]
