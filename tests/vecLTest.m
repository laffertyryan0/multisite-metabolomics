function tests = vecLTest
addpath("../src")
tests = functiontests(localfunctions);
end

% Check answer for a simple example
function testCompareKnown(testCase)
    mat = [[0 5 5 5];
           [1 0 5 5];
           [2 4 0 5];
           [3 5 6 0]];
    v = vecL(mat);
    verifyEqual(testCase,v,(1:6)');
end

