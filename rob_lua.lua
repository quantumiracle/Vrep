get_configuration_tree=function(inInts, inFloats, inStrings, inBuffer)
    local model_base = sim.getObjectHandle('redundantRobot')
    rawBufferHandle = sim.getConfigurationTree(model_base)
    return {rawBufferHandle}, {}, {}, ''
end

get_endpoint_position=function(inInts, inFloats, inStrings, inBuffer)
    -- get the endpoint position as a float vector
    endpoint_handle = sim.getObjectHandle('redundantRob_manipSphere')
    rob_handle = sim.getObjectHandle('redundantRobot')
    position = sim.getObjectPosition(endpoint_handle,rob_handle)
    return {}, position, {}, ''
end

set_endpoint_position=function(inInts, inFloats, inStrings, inBuffer)
    -- the set object position takes in the ObjectHandle and a 3d vector
    -- number result=sim.setObjectPosition(number objectHandle,number relativeToObjectHandle,table_3 position)
    -- relativeToObjectHandle: indicates relative to which reference frame we want the matrix.
    -- Specify -1 to retrieve the absolute transformation matrix,
    -- sim.handle_parent to retrieve the transformation matrix relative to the object's parent,
    -- or an object handle relative to whose reference frame we want the transformation matrix.
    endpoint_handle = sim.getObjectHandle('redundantRob_manipSphere')
    rob_handle = sim.getObjectHandle('redundantRobot')

    -- create a table (which is basically a list), table_3 = list of 3 elements
    goal_position = {}

    -- check if the input float is a 3d vector
    if #inFloats ~= 3 then return -2 end

    -- for i in np.linspace(1, len(inFloats), steps=1)
    for i=1,#inFloats,1 do
        goal_position[i] = inFloats[i]
    end
    success = sim.setObjectPosition(endpoint_handle,rob_handle,goal_position)
    return {success}, {}, {}, ''
end

set_hand_open=function(inInts, inFloats, inStrings, inBuffer)
    -- accept an int, set open the hand if the int is 1, else, close the hand
    hand_force = 50
    finger1_handle = sim.getObjectHandle('redundantRob_finger1')
    finger2_handle = sim.getObjectHandle('redundantRob_finger2')
    hand_action = inInts[1]
    if hand_action == 1 then
        -- 1 is open hand
        sim.setJointForce(finger1_handle,hand_force)
        sim.setJointForce(finger2_handle,hand_force)
    else
        sim.setJointForce(finger1_handle,-1*hand_force)
        sim.setJointForce(finger2_handle,-1*hand_force)
    end
    return {1},{},{},''
end
