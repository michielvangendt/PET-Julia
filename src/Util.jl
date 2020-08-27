include("LineOfResponse.jl")

# Transform event to main plane
function transform_event(lor::LOR, plane::Plane)
    # Transform event so we can use the same code for different main planes.
    if plane == x_plane
        x1 = lor.P1.x
        x2 = lor.P2.x
        y1 = lor.P1.y
        y2 = lor.P2.y
        z1 = lor.P1.z
        z2 = lor.P2.z
    elseif plane == y_plane
        x1 = lor.P1.y
        x2 = lor.P2.y
        y1 = lor.P1.x
        y2 = lor.P2.x
        z1 = lor.P1.z
        z2 = lor.P2.z
    else
        x1 = lor.P1.z
        x2 = lor.P2.z
        y1 = lor.P1.y
        y2 = lor.P2.y
        z1 = lor.P1.x
        z2 = lor.P2.x
    end

    event_ret = Event(LOR(Point(x1, y1, z1), Point(x2, y2, z2), lor.timeDiff), plane)

    return event_ret
end

function read_3D(filename)
	DIMX = 512
	DIMY = 512
	DIMZ = 512

	fs = open("Data/" * filename,"r")
	array_length = stat("Data/" * filename).size ÷ (4*7)
	lor_type = Ref{LOR}()

	events = Array{Opt_Event}(undef, array_length)

	for i = 1:array_length
		lor = getindex(read!(fs, lor_type))
		
		x1 = lor.P1.x
		x2 = lor.P2.x
		y1 = lor.P1.y
		y2 = lor.P2.y
		z1 = lor.P1.z
		z2 = lor.P2.z

		if abs(x2-x1) > abs(y2-y1) && abs(x2-x1) > abs(z2-z1) # x-plane
			x1 = lor.P1.x
			x2 = lor.P2.x
			y1 = lor.P1.y
			y2 = lor.P2.y
			z1 = lor.P1.z
			z2 = lor.P2.z
			
			k_y = (y2 - y1) / (x2 - x1)
			k_z = (z2 - z1) / (x2 - x1)
			
			y_min_base = 0.0f0
			if k_y >= 0
				y_min_base = y1 - k_y*(DIMX/2 + 1 + x1) + DIMY/2
			else
				y_min_base = y1 - k_y*(DIMX/2 + x1) + DIMY/2
			end
			
			z_min_base = 0.0f0
			if k_z >= 0
				z_min_base = z1 - k_z*(DIMX/2 + 1 + x1) + DIMZ/2
			else
				z_min_base = z1 - k_z*(DIMX/2 + x1) + DIMZ/2
			end
			
			dconv = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
			
			e = Opt_Event(x1, x2, k_y, k_z, y_min_base, z_min_base, dconv, x_plane)
			events[i] = e
		elseif abs(y2-y1) > abs(x2-x1) && abs(y2-y1) > abs(z2-z1) # y-plane
			x1 = lor.P1.y
			x2 = lor.P2.y
			y1 = lor.P1.x
			y2 = lor.P2.x
			z1 = lor.P1.z
			z2 = lor.P2.z
		
			k_y = (y2 - y1) / (x2 - x1)
			k_z = (z2 - z1) / (x2 - x1)
			
			y_min_base = 0.0f0
			if k_y >= 0
				y_min_base = y1 - k_y*(DIMX/2 + 1 + x1) + DIMY/2
			else
				y_min_base = y1 - k_y*(DIMX/2 + x1) + DIMY/2
			end
			
			z_min_base = 0.0f0
			if k_z >= 0
				z_min_base = z1 - k_z*(DIMX/2 + 1 + x1) + DIMZ/2
			else
				z_min_base = z1 - k_z*(DIMX/2 + x1) + DIMZ/2
			end
			
			dconv = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
			
			e = Opt_Event(x1, x2, k_y, k_z, y_min_base, z_min_base, dconv, y_plane)
			events[i] = e
		elseif abs(z2-z1) > abs(x2-x1) && abs(z2-z1) > abs(y2-y1) # z-plane
			x1 = lor.P1.z
			x2 = lor.P2.z
			y1 = lor.P1.y
			y2 = lor.P2.y
			z1 = lor.P1.x
			z2 = lor.P2.x

			k_y = (y2 - y1) / (x2 - x1)
			k_z = (z2 - z1) / (x2 - x1)
			
			y_min_base = 0.0f0
			if k_y >= 0
				y_min_base = y1 - k_y*(DIMX/2 + 1 + x1) + DIMY/2
			else
				y_min_base = y1 - k_y*(DIMX/2 + x1) + DIMY/2
			end
			
			z_min_base = 0.0f0
			if k_z >= 0
				z_min_base = z1 - k_z*(DIMX/2 + 1 + x1) + DIMZ/2
			else
				z_min_base = z1 - k_z*(DIMX/2 + x1) + DIMZ/2
			end
			
			dconv = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
			
			e = Opt_Event(x1, x2, k_y, k_z, y_min_base, z_min_base, dconv, z_plane)
			events[i] = e
		else
			e = Opt_Event(0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, undef_plane)
			events[i] = e
		end
	end

	println("Total number of events: ", array_length)

	return events, array_length
end