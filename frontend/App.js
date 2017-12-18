import React, { Component } from 'react';
import { AppRegistry } from 'react-native';
import Record from './components/Record';

export default class Voxuoso extends Component {
	render() {
		return (
			<Record/>
    );
  }
}

AppRegistry.registerComponent('Voxuoso', () => Voxuoso);