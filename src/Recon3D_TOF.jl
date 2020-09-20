using CUDA

include("LineOfResponse.jl")
include("Util.jl")
include("Recon3D.jl")


struct TOF_event
	event::Event
	time_diff::Float32
	TOF_kernel::CuDeviceTexture{Float32, 1, CUDA.AS.Global}
end


function create_tof_kernel()
	tofres = 390

	fwhm = tofres*0.299792458/2.0 #  speed of light mm/ps
	sigma = fwhm/(2*sqrt(2*log(2)))
	len = floor(Integer, 5*sigma)

	kernel = zeros(Float32, len)

	sum = 0.0

	for i = 1:len
		kernel[i] = (1/(sqrt(2*pi)*sigma))*exp(-(i*i)/(2*sigma*sigma))
	    sum += kernel[i]
	end
	for i = 1:len
	    kernel[i] /= 2*sum;
	end

	return kernel
end

@inline function get_corr_factor(kernel, mean, x)
	idx = abs( floor(Integer, (mean-x)) )

	if idx < 124
		return kernel[idx+1]
	end
	return 0.0f0
end

@inline function forward_project(thread_i, TOF_event::TOF_event, image, DIMX, DIMY, DIMZ)
	block_size = Int32(4)
	one = Int32(1)

	event = TOF_event.event

    DIMX_event, DIMY_event, DIMZ_event = transform_dimensions(event.mainPlane, DIMX, DIMY, DIMZ)

    nr_of_iterations = fld1(DIMX_event - (thread_i-one), blockDim().x)
	slices = @cuDynamicSharedMem(Slice, DIMX_event)
    value = 0.0f0

    for i = one:nr_of_iterations
        t = Int32(thread_i + (i-one)*blockDim().x)

        # Ray tracing
        Y_min, Y_plus, Z_min, Z_plus, l_min_min, l_min_plus, l_plus_min, l_plus_plus = ray_tracing(event, t, DIMX_event, DIMY_event, DIMZ_event)

		x1 = event.lor.P1.x
		x2 = event.lor.P2.x
		y1 = event.lor.P1.y
		y2 = event.lor.P2.y
		z1 = event.lor.P1.z
		z2 = event.lor.P2.z

		dconv = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
		current_x = (t - DIMX_event/2.0f0 - 1.0f0)*3.125f0
		loc_on_line = (current_x - x1)/(x2 - x1) * dconv

		mean_loc_on_line = 0.5f0*(dconv-0.299792458f0*TOF_event.time_diff);
		tof_kernel = TOF_event.TOF_kernel
		corr_factor = get_corr_factor(tof_kernel, mean_loc_on_line, loc_on_line)

		i1 = calculate_index(perm(event.mainPlane, t, Y_min + 1, Z_min + 1)..., DIMX, DIMY, DIMZ, 4)
        i2 = calculate_index(perm(event.mainPlane, t, Y_min + 1, Z_plus + 1)..., DIMX, DIMY, DIMZ, 4)
        i3 = calculate_index(perm(event.mainPlane, t, Y_plus + 1, Z_min + 1)..., DIMX, DIMY, DIMZ, 4)
        i4 = calculate_index(perm(event.mainPlane, t, Y_plus + 1, Z_plus + 1)..., DIMX, DIMY, DIMZ, 4)

        if Y_min >= 0 && Z_min >=0 && Y_plus < DIMY_event && Z_plus < DIMZ_event
			l_min_min *= corr_factor
			l_min_plus *= corr_factor
			l_plus_min *= corr_factor
			l_plus_plus *= corr_factor

            @inbounds value += image[i1] * l_min_min
            @inbounds value += image[i2] * l_min_plus
            @inbounds value += image[i3] * l_plus_min
            @inbounds value += image[i4] * l_plus_plus

			slices[t] = Slice(true, i1, i2, i3, i4, l_min_min, l_min_plus, l_plus_min, l_plus_plus)
		else
			slices[t] = Slice(false, i1, i2, i3, i4, l_min_min, l_min_plus, l_plus_min, l_plus_plus)
        end
	end

    # Forward project

    # Calculate total ray length
    total_value = reduce_block(+, value, 0.0f0)

	# Share total length across all threads in the block
	shared_total_value = @cuStaticSharedMem(Float32, 1)
	if thread_i == 1
		shared_total_value[1] = total_value
	end
	CUDA.sync_threads()
	total_value = shared_total_value[1]

	return total_value, nr_of_iterations, slices
end

@inline function calc_corr_factor(total_value, event::TOF_event)
	return total_value
end

function tof_reconstruct3D(events, sensmap, DIMX, DIMY, DIMZ, recon_iters)
	c_events = CUDA.CuArray(events);

	c_image = CUDA.ones(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

	c_sensmap = CUDA.CuArray(sensmap);

	n_threads = min(max(DIMX, DIMY, DIMZ), 128)
	n_blocks = length(c_events)
	n_shmem = max(DIMX, DIMY, DIMZ)*sizeof(Slice)

	for k = 1:recon_iters
		c_corr = CUDA.zeros(Float32, calculate_length(DIMX, DIMY, DIMZ, 4));

		@cuda blocks=n_blocks threads=n_threads shmem=n_shmem gpu_kernel(c_events, c_image, c_corr, Int32(DIMX), Int32(DIMY), Int32(DIMZ))

		c_image = c_image .* c_corr ./ c_sensmap
	end

	image = Array(c_image)

	return image
end