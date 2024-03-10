import styled, {keyframes} from "styled-components";

const spinningAnimation = keyframes`
0% {transform: rotate(0deg)}
100% {transform: rotate(359deg)}
`

export const Loading = styled.img`
width: 100px;
height: auto;
display: block;
position: absolute;
left: 50%;
margin: 2em 0;
animation-name: ${spinningAnimation};
animation-duration: 1.5s;
animation-iteration-count: infinite;
animation-timing-function: linear;
filter: invert(45%) sepia(12%) saturate(0%) hue-rotate(209deg) brightness(100%) contrast(93%);
`;