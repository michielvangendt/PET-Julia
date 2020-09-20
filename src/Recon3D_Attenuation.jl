# using CUDA

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")

struct Att_event
	event::Event
	acf::Float32
end


function gpu_calc_att(events::CuDeviceArray{Event}, att_events::CuDeviceArray{Att_event}, att_map::CuDeviceArray{T,1}, DIMX::Int32, DIMY::Int32, DIMZ::Int32) where {T}
    thread_i = threadIdx().x
    index = blockIdx().x

	event = get_element(thread_i, index, events)
	total_value, nr_of_iterations, slices = forward_project(thread_i, event, att_map,  DIMX, DIMY, DIMZ)

    if thread_i == 1
        acf = CUDA.exp(-total_value*3.125f0)
		att_events[index] = Att_event(event, acf)
    end

    return nothing
end

# Reuse existing forward projection code
@inline @inbounds function forward_project(thread_i, att_event::Att_event, image, DIMX, DIMY, DIMZ)
    forward_project(thread_i, att_event.event, image, DIMX, DIMY, DIMZ)
end

# We redefine the calc_corr_factor function, but ths time for an Att_event type
@inline function calc_corr_factor(total_value, att_event::Att_event)
    return total_value*att_event.acf
end

function att_reconstruct3D(events, sensmap, attmap, DIMX, DIMY, DIMZ, recon_iters)
	n_threads = min(max(DIMX, DIMY, DIMZ), 128)
	n_blocks = length(events)
	n_shmem = max(DIMX, DIMY, DIMZ)*sizeof(Slice)

	c_att_map = CUDA.CuArray(attmap);
	c_events = CUDA.CuArray(events);
	c_att_events = CUDA.CuArray(Array{Att_event}(undef, length(events)));

    # Calculate attenuation correction factor
	@cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_calc_att(c_events, c_att_events, c_att_map, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

	c_sensmap = CUDA.CuArray(sensmap);
	c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

	for k = 1:recon_iters
		c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

		# We call the gpu_kernel function from Recon3D.jl
        # We do not need to rewrite the kernel, but Julia will substitute the forward_project and
        # calc_corr_factor function specified for the Att_event type automatically.
		@cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_att_events, c_image, c_corr, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

		c_image = c_image .* c_corr ./ c_sensmap
	end

	image = Array(c_image)

	return image
end