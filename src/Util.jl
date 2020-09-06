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
	fs = open("Data/" * filename,"r")
	array_length = Int(stat("Data/" * filename).size / (4*7))
	lor_type = Ref{LOR}()

	events = Array{Event}(undef, array_length)

	for i = 1:array_length
		lor = getindex(read!(fs, lor_type))

		if abs(lor.P2.x-lor.P1.x) > abs(lor.P2.y-lor.P1.y) && abs(lor.P2.x-lor.P1.x) > abs(lor.P2.z-lor.P1.z) # x-plane
			e = transform_event(lor, x_plane)
			events[i] = e
		elseif abs(lor.P2.y-lor.P1.y) > abs(lor.P2.x-lor.P1.x) && abs(lor.P2.y-lor.P1.y) > abs(lor.P2.z-lor.P1.z) # y-plane
			e = transform_event(lor, y_plane)
			events[i] = e
		elseif abs(lor.P2.z-lor.P1.z) > abs(lor.P2.x-lor.P1.x) && abs(lor.P2.z-lor.P1.z) > abs(lor.P2.y-lor.P1.y) # z-plane
			e = transform_event(lor, z_plane)
			events[i] = e
		else
			events[i] = Event(lor, undef_plane)
		end
	end

	println("Total number of events: ", array_length)

	return events, array_length
end